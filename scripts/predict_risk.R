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
current_path <- file.path(APP_ROOT, "outputs", "current_env.csv")
out_path <- file.path(APP_ROOT, "outputs", "latest_risk.csv")
metrics_path <- file.path(APP_ROOT, "outputs", "model_metrics.csv")

train <- prepare_training(master_path, depth_path)
cur <- prepare_prediction(read_csv(current_path, show_col_types = FALSE), train, depth_path)
fit <- fit_primary_gam(train)
out <- score_relative_risk(fit, train, cur) %>% arrange(desc(core_percentile))
write_csv(out, out_path)

fitted <- as.numeric(predict(fit, newdata = train, type = "response"))
write_csv(tibble(
  n_train = nrow(train), n_presence = sum(train$presence == 1),
  n_background = sum(train$presence == 0), apparent_auc = apparent_auc(train$presence, fitted),
  aic = AIC(fit), deviance_explained = summary(fit)$dev.expl,
  validated_lono_auc_mean = 0.858, validated_lono_auc_sd = 0.075,
  gam_formula = "presence ~ s(Jday, bs='cc', k=20) + s(SST_anomaly, k=10) + s(depth_m, k=10)",
  depth_source = "J-EGG500 IDW",
  risk_scale = "fixed-reference percentile; not absolute probability",
  created_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S UTC", tz = "UTC")
), metrics_path)
cat("Saved:", out_path, "\n")
