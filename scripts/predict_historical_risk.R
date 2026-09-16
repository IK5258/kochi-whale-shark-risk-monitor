get_root <- function() {
  root <- Sys.getenv("APP_ROOT", unset = "")
  if (nzchar(root)) return(normalizePath(root))
  if (dir.exists("data") && dir.exists("scripts")) return(normalizePath(getwd()))
  file_arg <- grep("--file=", commandArgs(FALSE), value = TRUE)
  normalizePath(file.path(dirname(sub("--file=", "", file_arg[1])), ".."))
}

APP_ROOT <- get_root()
source(file.path(APP_ROOT, "scripts", "model_core.R"))
master_path <- file.path(APP_ROOT, "data", "WhaleShark_env_master.csv")
depth_path <- file.path(APP_ROOT, "data", "net_depth_jegg500.csv")
hist_env_path <- file.path(APP_ROOT, "outputs", "historical_env_daily.csv")
out_daily_path <- file.path(APP_ROOT, "outputs", "historical_risk_daily.csv")
out_monthly_path <- file.path(APP_ROOT, "outputs", "historical_risk_monthly.csv")
out_yearly_path <- file.path(APP_ROOT, "outputs", "historical_risk_yearly.csv")

input_path <- if (file.exists(hist_env_path)) hist_env_path else out_daily_path
if (!file.exists(input_path)) stop("No historical environment input is available")
train <- prepare_training(master_path, depth_path)
cur <- read_csv(input_path, show_col_types = FALSE)
cur <- cur %>% select(-any_of(c(
  "maxent_suitability", "maxent_source", "integrated_risk",
  "integrated_percentile_today", "integrated_risk_class", "integrated_formula"
)))
cur <- prepare_prediction(cur, train, depth_path)
fit <- fit_primary_gam(train)
cur <- score_relative_risk(fit, train, cur)

if (!("date" %in% names(cur)) && "Date" %in% names(cur)) cur$date <- cur$Date
if (!("target_date" %in% names(cur))) cur$target_date <- cur$date
if (!("net_name" %in% names(cur))) cur$net_name <- paste0("NetID ", cur$NetID)
if (!("net_label" %in% names(cur))) cur$net_label <- paste0(cur$net_name, " (NetID ", cur$NetID, ")")

daily <- cur %>%
  mutate(
    date = as.Date(date), target_date = as.Date(target_date),
    Year = as.integer(format(date, "%Y")), Month = as.integer(format(date, "%m")),
    Day = as.integer(format(date, "%d")), period_label = as.character(date),
    forecast_ocean_mode = "historical_oisst",
    ocean_forecast_source = "NOAA OISST historical"
  ) %>%
  arrange(date, desc(core_percentile))

monthly <- daily %>%
  group_by(Year, Month, NetID, Latitude, Longitude, depth_m, GEBCO_old_m,
           depth_JEGG500_IDW_m, JEGG_quality, net_name, net_label) %>%
  summarise(
    date = as.Date(sprintf("%04d-%02d-15", first(Year), first(Month))),
    target_date = date, Jday = round(mean(Jday, na.rm = TRUE)),
    SST = mean(SST, na.rm = TRUE), SST_climatology = mean(SST_climatology, na.rm = TRUE),
    SST_anomaly = mean(SST_anomaly, na.rm = TRUE), core_risk = mean(core_risk, na.rm = TRUE),
    core_percentile = mean(core_percentile, na.rm = TRUE), .groups = "drop"
  ) %>%
  mutate(
    core_risk_class = risk_class_from_percentile(core_percentile),
    relative_risk_percentile = core_percentile, relative_risk_class = core_risk_class,
    period_label = sprintf("%04d-%02d", Year, Month),
    model_main = "GAM: cyclic Jday + SST anomaly + J-EGG500 IDW depth",
    forecast_ocean_mode = "historical_oisst_monthly_mean",
    ocean_forecast_source = "NOAA OISST historical",
    note = "Monthly mean of fixed-reference relative percentiles; not an absolute probability."
  ) %>% arrange(Year, Month, desc(core_percentile))

yearly <- daily %>%
  group_by(Year, NetID, Latitude, Longitude, depth_m, GEBCO_old_m,
           depth_JEGG500_IDW_m, JEGG_quality, net_name, net_label) %>%
  summarise(
    date = as.Date(sprintf("%04d-07-01", first(Year))), target_date = date,
    Jday = round(mean(Jday, na.rm = TRUE)), SST = mean(SST, na.rm = TRUE),
    SST_climatology = mean(SST_climatology, na.rm = TRUE),
    SST_anomaly = mean(SST_anomaly, na.rm = TRUE), core_risk = mean(core_risk, na.rm = TRUE),
    core_percentile = mean(core_percentile, na.rm = TRUE), .groups = "drop"
  ) %>%
  mutate(
    Month = NA_integer_, core_risk_class = risk_class_from_percentile(core_percentile),
    relative_risk_percentile = core_percentile, relative_risk_class = core_risk_class,
    period_label = as.character(Year),
    model_main = "GAM: cyclic Jday + SST anomaly + J-EGG500 IDW depth",
    forecast_ocean_mode = "historical_oisst_yearly_mean",
    ocean_forecast_source = "NOAA OISST historical",
    note = "Yearly mean of fixed-reference relative percentiles; not an absolute probability."
  ) %>% arrange(Year, desc(core_percentile))

write_csv(daily, out_daily_path)
write_csv(monthly, out_monthly_path)
write_csv(yearly, out_yearly_path)
message("Saved historical daily, monthly, and yearly relative-risk outputs")
