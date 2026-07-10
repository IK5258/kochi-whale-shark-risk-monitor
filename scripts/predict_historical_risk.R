
suppressPackageStartupMessages({
  library(mgcv)
  library(readr)
  library(dplyr)
  library(tibble)
})

get_root <- function() {
  env_root <- Sys.getenv("APP_ROOT", unset = NA)
  if (!is.na(env_root) && nzchar(env_root)) return(normalizePath(env_root))

  if (dir.exists("data") && dir.exists("scripts")) return(normalizePath(getwd()))

  cmd <- commandArgs(FALSE)
  file_arg <- grep("--file=", cmd, value = TRUE)

  if (length(file_arg) > 0) {
    script_path <- normalizePath(sub("--file=", "", file_arg[1]))
    return(normalizePath(file.path(dirname(script_path), "..")))
  }

  return(normalizePath(getwd()))
}

APP_ROOT <- get_root()

master_path <- file.path(APP_ROOT, "data", "WhaleShark_env_master.csv")
hist_env_path <- file.path(APP_ROOT, "outputs", "historical_env_daily.csv")

out_daily_path <- file.path(APP_ROOT, "outputs", "historical_risk_daily.csv")
out_monthly_path <- file.path(APP_ROOT, "outputs", "historical_risk_monthly.csv")
out_yearly_path <- file.path(APP_ROOT, "outputs", "historical_risk_yearly.csv")


rename_first <- function(df, candidates, new_name, required = FALSE) {
  if (new_name %in% names(df)) return(df)

  hit <- candidates[candidates %in% names(df)]

  if (length(hit) > 0) {
    names(df)[names(df) == hit[1]] <- new_name
    return(df)
  }

  if (required) {
    stop(paste0("Missing required column for ", new_name, ". Tried: ", paste(candidates, collapse = ", ")))
  }

  return(df)
}


risk_class_from_percentile <- function(p) {
  dplyr::case_when(
    is.na(p) ~ "Unknown",
    p >= 0.90 ~ "Very high",
    p >= 0.75 ~ "High",
    p >= 0.50 ~ "Moderate",
    TRUE ~ "Low"
  )
}


if (!file.exists(master_path)) stop(paste("Missing:", master_path))
if (!file.exists(hist_env_path)) stop(paste("Missing:", hist_env_path))

df <- read_csv(master_path, show_col_types = FALSE)
cur <- read_csv(hist_env_path, show_col_types = FALSE)

df <- rename_first(df, c("Presence", "presence", "pa", "PA", "occurrence"), "presence", required = TRUE)
df <- rename_first(df, c("Jday", "jday", "Julian_day", "julian_day"), "Jday", required = TRUE)
df <- rename_first(df, c("SST", "sst", "SST_C", "sst_C"), "SST", required = TRUE)
df <- rename_first(df, c("depth_m", "Depth_m", "depth", "Depth", "bathymetry_m"), "depth_m", required = TRUE)

cur <- rename_first(cur, c("NetID", "net_id", "net"), "NetID", required = TRUE)
cur <- rename_first(cur, c("Latitude", "latitude", "lat", "Lat"), "Latitude", required = TRUE)
cur <- rename_first(cur, c("Longitude", "longitude", "lon", "Lon"), "Longitude", required = TRUE)
cur <- rename_first(cur, c("Jday", "jday", "Julian_day", "julian_day"), "Jday", required = TRUE)
cur <- rename_first(cur, c("SST", "sst", "SST_C", "sst_C"), "SST", required = TRUE)
cur <- rename_first(cur, c("depth_m", "Depth_m", "depth", "Depth", "bathymetry_m"), "depth_m", required = TRUE)

df <- df %>%
  mutate(
    presence = as.integer(presence),
    Jday = as.numeric(Jday),
    SST = as.numeric(SST),
    depth_m = abs(as.numeric(depth_m))
  ) %>%
  filter(
    presence %in% c(0, 1),
    !is.na(Jday),
    !is.na(SST),
    !is.na(depth_m)
  )

cur <- cur %>%
  mutate(
    NetID = as.character(NetID),
    Jday = as.numeric(Jday),
    SST = as.numeric(SST),
    depth_m = abs(as.numeric(depth_m)),
    Latitude = as.numeric(Latitude),
    Longitude = as.numeric(Longitude)
  )

if (any(is.na(cur$SST))) {
  cur$SST[is.na(cur$SST)] <- median(df$SST, na.rm = TRUE)
}

if (any(is.na(cur$depth_m))) {
  cur$depth_m[is.na(cur$depth_m)] <- median(df$depth_m, na.rm = TRUE)
}

df$Jday <- pmin(pmax(df$Jday, 1), 366)
cur$Jday <- pmin(pmax(cur$Jday, 1), 366)

gam_fit <- gam(
  presence ~ s(Jday, bs = "cc", k = 12) + s(SST, k = 8) + s(depth_m, k = 8),
  data = df,
  family = binomial,
  method = "REML",
  knots = list(Jday = c(0.5, 366.5))
)

df$core_fitted <- as.numeric(predict(gam_fit, newdata = df, type = "response"))
cur$core_risk <- as.numeric(predict(gam_fit, newdata = cur, type = "response"))

core_ecdf <- ecdf(df$core_fitted)
cur$core_percentile <- as.numeric(core_ecdf(cur$core_risk))
cur$core_risk_class <- risk_class_from_percentile(cur$core_percentile)

df_mx <- df %>%
  mutate(
    Jday_sin = sin(2 * pi * Jday / 366),
    Jday_cos = cos(2 * pi * Jday / 366)
  ) %>%
  select(presence, Jday_sin, Jday_cos, SST, depth_m) %>%
  filter(complete.cases(.))

cur_mx <- cur %>%
  mutate(
    Jday_sin = sin(2 * pi * Jday / 366),
    Jday_cos = cos(2 * pi * Jday / 366)
  )

maxent_source <- "glm_maxent_like"

if (requireNamespace("maxnet", quietly = TRUE)) {
  maxent_source <- "maxnet_lq"

  mx_x <- df_mx %>% select(Jday_sin, Jday_cos, SST, depth_m)
  mx_p <- df_mx$presence

  mx_fit <- maxnet::maxnet(
    p = mx_p,
    data = mx_x,
    f = maxnet::maxnet.formula(mx_p, mx_x, classes = "lq"),
    regmult = 1
  )

  cur$maxent_suitability <- as.numeric(
    predict(
      mx_fit,
      newdata = cur_mx %>% select(Jday_sin, Jday_cos, SST, depth_m),
      type = "cloglog"
    )
  )

} else {
  mx_fit <- glm(
    presence ~ Jday_sin + Jday_cos + poly(SST, 2, raw = TRUE) + poly(depth_m, 2, raw = TRUE),
    data = df_mx,
    family = binomial
  )

  cur$maxent_suitability <- as.numeric(
    predict(mx_fit, newdata = cur_mx, type = "response")
  )
}

cur <- cur %>%
  mutate(
    integrated_risk = core_risk,
    integrated_percentile_today = core_percentile,
    integrated_risk_class = core_risk_class,
    model_main = "GAM: Jday + SST + depth",
    maxent_source = maxent_source,
    forecast_ocean_mode = "historical_oisst",
    ocean_forecast_source = "NOAA OISST historical",
    note = "Historical risk is based on OISST, Jday, and GEBCO depth. Kuroshio and upwelling contexts are not included."
  )

if (!("date" %in% names(cur)) && "Date" %in% names(cur)) {
  cur$date <- cur$Date
}

if (!("target_date" %in% names(cur))) {
  cur$target_date <- cur$date
}

cur <- cur %>%
  mutate(
    date = as.Date(date),
    target_date = as.Date(target_date),
    Year = as.integer(format(date, "%Y")),
    Month = as.integer(format(date, "%m")),
    Day = as.integer(format(date, "%d")),
    period_label = as.character(date)
  )

daily <- cur %>%
  arrange(date, desc(integrated_risk))

monthly <- daily %>%
  group_by(Year, Month, NetID, Latitude, Longitude, depth_m, net_name, net_label) %>%
  summarise(
    date = as.Date(sprintf("%04d-%02d-15", first(Year), first(Month))),
    target_date = date,
    Jday = round(mean(Jday, na.rm = TRUE)),
    SST = mean(SST, na.rm = TRUE),
    core_risk = mean(core_risk, na.rm = TRUE),
    core_percentile = mean(core_percentile, na.rm = TRUE),
    maxent_suitability = mean(maxent_suitability, na.rm = TRUE),
    integrated_risk = mean(integrated_risk, na.rm = TRUE),
    integrated_percentile_today = mean(integrated_percentile_today, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    core_risk_class = risk_class_from_percentile(core_percentile),
    integrated_risk_class = risk_class_from_percentile(integrated_percentile_today),
    period_label = sprintf("%04d-%02d", Year, Month),
    model_main = "GAM: Jday + SST + depth",
    maxent_source = maxent_source,
    forecast_ocean_mode = "historical_oisst_monthly_mean",
    ocean_forecast_source = "NOAA OISST historical",
    note = "Monthly historical risk is the mean of daily historical risks."
  ) %>%
  arrange(Year, Month, desc(integrated_risk))

yearly <- daily %>%
  group_by(Year, NetID, Latitude, Longitude, depth_m, net_name, net_label) %>%
  summarise(
    date = as.Date(sprintf("%04d-07-01", first(Year))),
    target_date = date,
    Jday = round(mean(Jday, na.rm = TRUE)),
    SST = mean(SST, na.rm = TRUE),
    core_risk = mean(core_risk, na.rm = TRUE),
    core_percentile = mean(core_percentile, na.rm = TRUE),
    maxent_suitability = mean(maxent_suitability, na.rm = TRUE),
    integrated_risk = mean(integrated_risk, na.rm = TRUE),
    integrated_percentile_today = mean(integrated_percentile_today, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    Month = NA_integer_,
    core_risk_class = risk_class_from_percentile(core_percentile),
    integrated_risk_class = risk_class_from_percentile(integrated_percentile_today),
    period_label = as.character(Year),
    model_main = "GAM: Jday + SST + depth",
    maxent_source = maxent_source,
    forecast_ocean_mode = "historical_oisst_yearly_mean",
    ocean_forecast_source = "NOAA OISST historical",
    note = "Yearly historical risk is the mean of daily historical risks."
  ) %>%
  arrange(Year, desc(integrated_risk))

write_csv(daily, out_daily_path)
write_csv(monthly, out_monthly_path)
write_csv(yearly, out_yearly_path)

message("Saved: ", out_daily_path)
message("Saved: ", out_monthly_path)
message("Saved: ", out_yearly_path)

print(daily %>% select(date, NetID, net_name, Jday, SST, depth_m, core_risk, integrated_risk_class) %>% head(20))
