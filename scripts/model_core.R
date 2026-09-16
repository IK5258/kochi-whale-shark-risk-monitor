suppressPackageStartupMessages({
  library(mgcv)
  library(readr)
  library(dplyr)
  library(tibble)
})

rename_first <- function(df, candidates, new_name, required = FALSE) {
  if (new_name %in% names(df)) return(df)
  hit <- candidates[candidates %in% names(df)]
  if (length(hit) > 0) {
    names(df)[names(df) == hit[1]] <- new_name
    return(df)
  }
  if (required) stop(paste0("Missing required column for ", new_name))
  df
}

normalize_net_id <- function(x) {
  x <- trimws(as.character(x))
  sub("\\.0+$", "", x)
}

risk_class_from_percentile <- function(p) {
  case_when(
    is.na(p) ~ "Unknown",
    p >= 0.90 ~ "Very high",
    p >= 0.75 ~ "High",
    p >= 0.50 ~ "Moderate",
    TRUE ~ "Low"
  )
}

read_depth_lookup <- function(depth_lookup_path) {
  read_csv(depth_lookup_path, show_col_types = FALSE) %>%
    transmute(
      NetID = normalize_net_id(NetID),
      Latitude_final = as.numeric(Latitude_final),
      Longitude_final = as.numeric(Longitude_final),
      GEBCO_old_m = as.numeric(GEBCO_old_m),
      depth_JEGG500_IDW_m = abs(as.numeric(depth_JEGG500_IDW_m)),
      JEGG_quality = as.character(JEGG_quality)
    ) %>%
    distinct(NetID, .keep_all = TRUE)
}

prepare_training <- function(master_path, depth_lookup_path) {
  df <- read_csv(master_path, show_col_types = FALSE)
  depth <- read_depth_lookup(depth_lookup_path)
  df <- rename_first(df, c("Presence", "pa", "PA", "occurrence"), "presence", TRUE)
  df <- rename_first(df, c("jday", "Julian_day", "julian_day"), "Jday", TRUE)
  df <- rename_first(df, c("sst", "SST_C", "sst_C"), "SST", TRUE)
  df <- rename_first(df, c("sst_anomaly", "SST_anom"), "SST_anomaly", TRUE)
  df <- rename_first(df, c("net_id", "net"), "NetID", TRUE)

  df %>%
    mutate(
      NetID = normalize_net_id(NetID),
      presence = as.integer(presence),
      Jday = pmin(pmax(as.numeric(Jday), 1), 366),
      SST = as.numeric(SST),
      SST_anomaly = as.numeric(SST_anomaly)
    ) %>%
    left_join(depth, by = "NetID", relationship = "many-to-one") %>%
    mutate(depth_m = depth_JEGG500_IDW_m) %>%
    filter(presence %in% c(0, 1), complete.cases(Jday, SST_anomaly, depth_m))
}

add_sst_anomaly <- function(cur, train) {
  clim_value <- if ("SST_jday_mean" %in% names(train)) {
    as.numeric(train$SST_jday_mean)
  } else {
    as.numeric(train$SST) - as.numeric(train$SST_anomaly)
  }
  clim <- tibble(Jday = as.integer(train$Jday), clim = clim_value) %>%
    filter(complete.cases(Jday, clim)) %>%
    group_by(Jday) %>%
    summarise(clim = mean(clim), .groups = "drop")

  nearest_clim <- function(day) {
    dist <- pmin(abs(clim$Jday - day), 366 - abs(clim$Jday - day))
    mean(clim$clim[dist == min(dist)], na.rm = TRUE)
  }
  cur$SST_climatology <- vapply(cur$Jday, nearest_clim, numeric(1))
  cur$SST_anomaly <- as.numeric(cur$SST) - cur$SST_climatology
  cur
}

prepare_prediction <- function(cur, train, depth_lookup_path) {
  depth <- read_depth_lookup(depth_lookup_path)
  cur <- rename_first(cur, c("net_id", "net"), "NetID", TRUE)
  cur <- rename_first(cur, c("jday", "Julian_day", "julian_day"), "Jday", TRUE)
  cur <- rename_first(cur, c("sst", "SST_C", "sst_C"), "SST", TRUE)
  for (nm in c("Latitude_final", "Longitude_final", "GEBCO_old_m", "depth_JEGG500_IDW_m", "JEGG_quality")) {
    if (nm %in% names(cur)) cur[[nm]] <- NULL
  }

  cur <- cur %>%
    mutate(
      NetID = normalize_net_id(NetID),
      Jday = pmin(pmax(as.numeric(Jday), 1), 366),
      SST = as.numeric(SST)
    ) %>%
    left_join(depth, by = "NetID", relationship = "many-to-one") %>%
    mutate(
      Latitude = Latitude_final,
      Longitude = Longitude_final,
      depth_m = depth_JEGG500_IDW_m
    )
  if (any(is.na(cur$depth_m))) stop("Prediction rows contain NetID without J-EGG500 depth")
  if (any(is.na(cur$SST))) cur$SST[is.na(cur$SST)] <- median(train$SST, na.rm = TRUE)
  add_sst_anomaly(cur, train)
}

fit_primary_gam <- function(train) {
  gam(
    presence ~ s(Jday, bs = "cc", k = 20) + s(SST_anomaly, k = 10) + s(depth_m, k = 10),
    data = train,
    family = binomial,
    method = "REML",
    select = TRUE,
    knots = list(Jday = c(0.5, 366.5))
  )
}

score_relative_risk <- function(fit, train, cur) {
  train_fitted <- as.numeric(predict(fit, newdata = train, type = "response"))
  cur$core_risk <- as.numeric(predict(fit, newdata = cur, type = "response"))
  cur$core_percentile <- as.numeric(ecdf(train_fitted)(cur$core_risk))
  cur$core_risk_class <- risk_class_from_percentile(cur$core_percentile)
  cur$relative_risk_percentile <- cur$core_percentile
  cur$relative_risk_class <- cur$core_risk_class
  cur$model_main <- "GAM: cyclic Jday + SST anomaly + J-EGG500 IDW depth"
  cur$note <- "Relative percentile against the fixed training reference; not an absolute occurrence probability."
  cur
}

apparent_auc <- function(y, score) {
  ok <- complete.cases(y, score)
  y <- y[ok]
  score <- score[ok]
  n1 <- sum(y == 1)
  n0 <- sum(y == 0)
  if (n1 == 0 || n0 == 0) return(NA_real_)
  (sum(rank(score, ties.method = "average")[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
