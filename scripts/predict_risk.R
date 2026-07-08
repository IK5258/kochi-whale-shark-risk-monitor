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
current_path <- file.path(APP_ROOT, "outputs", "current_env.csv")
out_path <- file.path(APP_ROOT, "outputs", "latest_risk.csv")
metrics_path <- file.path(APP_ROOT, "outputs", "model_metrics.csv")

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

safe_rescale <- function(x, lo, hi) {
  y <- (x - lo) / (hi - lo)
  y <- pmin(pmax(y, 0), 1)
  return(y)
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
if (!file.exists(current_path)) stop(paste("Missing:", current_path))

df <- read_csv(master_path, show_col_types = FALSE)
cur <- read_csv(current_path, show_col_types = FALSE)

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

if ("upwelling_proxy" %in% names(cur)) {
  cur$upwelling_proxy <- as.numeric(cur$upwelling_proxy)

  if ("upwelling_proxy_divergence" %in% names(df)) {
    df$upwelling_proxy_divergence <- as.numeric(df$upwelling_proxy_divergence)
    q <- quantile(df$upwelling_proxy_divergence, probs = c(0.05, 0.95), na.rm = TRUE)
    lo <- as.numeric(q[1])
    hi <- as.numeric(q[2])
  } else if ("upwelling_proxy" %in% names(df)) {
    df$upwelling_proxy <- as.numeric(df$upwelling_proxy)
    q <- quantile(df$upwelling_proxy, probs = c(0.05, 0.95), na.rm = TRUE)
    lo <- as.numeric(q[1])
    hi <- as.numeric(q[2])
  } else {
    q <- quantile(cur$upwelling_proxy, probs = c(0.05, 0.95), na.rm = TRUE)
    lo <- as.numeric(q[1])
    hi <- as.numeric(q[2])
  }

  if (is.na(lo) || is.na(hi) || lo == hi) {
    cur$upwelling_context <- 0.5
  } else {
    cur$upwelling_context <- safe_rescale(cur$upwelling_proxy, lo, hi)
    cur$upwelling_context[is.na(cur$upwelling_context)] <- 0.5
  }

} else {
  cur$upwelling_proxy <- NA_real_
  cur$upwelling_context <- 0.5
}

cur <- cur %>%
  mutate(
    kuroshio_dist_km = NA_real_,
    kuroshio_context = NA_real_,
    integrated_risk = 0.85 * core_risk +
      0.15 * upwelling_context,
    integrated_formula = "0.85 core + 0.15 upwelling; kuroshio not implemented in v0.1"
  )

int_ecdf <- ecdf(cur$integrated_risk)
cur$integrated_percentile_today <- as.numeric(int_ecdf(cur$integrated_risk))
cur$integrated_risk_class <- risk_class_from_percentile(cur$integrated_percentile_today)

cur <- cur %>%
  mutate(
    model_main = "GAM: Jday + SST + depth",
    maxent_source = maxent_source,
    note = "Integrated risk is experimental. Core GAM risk should be treated as the main risk index. Kuroshio context is not implemented in v0.1."
  ) %>%
  arrange(desc(core_risk))

dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
write_csv(cur, out_path)

metrics <- tibble(
  n_train = nrow(df),
  n_presence = sum(df$presence == 1),
  n_background = sum(df$presence == 0),
  gam_formula = "presence ~ s(Jday, bs='cc') + s(SST) + s(depth_m)",
  maxent_source = maxent_source,
  integrated_formula = "0.85 core + 0.15 upwelling; kuroshio not implemented in v0.1",
  created_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S")
)

write_csv(metrics, metrics_path)

cat("Saved:", out_path, "\n")
print(
  cur %>%
    select(NetID, Jday, SST, depth_m, core_risk, upwelling_context, integrated_risk, integrated_risk_class) %>%
    head()
)
