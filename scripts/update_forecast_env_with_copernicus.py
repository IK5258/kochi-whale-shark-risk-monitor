from pathlib import Path
from datetime import datetime, timedelta, timezone
import os
import numpy as np
import pandas as pd
import xarray as xr
import copernicusmarine


APP_ROOT = Path(os.environ.get("APP_ROOT", Path(__file__).resolve().parents[1]))

FORECAST_ENV_PATH = APP_ROOT / "outputs" / "forecast_env.csv"
OUT_PATH = APP_ROOT / "outputs" / "forecast_env.csv"
RAW_DIR = APP_ROOT / "data" / "raw_copernicus_forecast"

FORECAST_DAYS_REAL = [0, 3, 7, 10]

LON_MIN = 131.5
LON_MAX = 136.0
LAT_MIN = 28.0
LAT_MAX = 34.5

TEMP_DATASETS = [
    "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
    "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
]

CUR_DATASETS = [
    "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
    "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
]

WCUR_DATASETS = [
    "cmems_mod_glo_phy-wcur_anfc_0.083deg_P1D-m",
    "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
]


def jst_today():
    run_date = os.environ.get("RUN_DATE", "")
    if run_date:
        return datetime.strptime(run_date, "%Y-%m-%d").date()
    return datetime.now(timezone(timedelta(hours=9))).date()


def find_coord_name(ds, candidates):
    for c in candidates:
        if c in ds.coords:
            return c
        if c in ds.dims:
            return c
    raise ValueError(f"Cannot find coordinate from: {candidates}")


def get_depth_name(ds):
    for c in ["depth", "depthu", "depthv", "depthw", "elevation"]:
        if c in ds.coords or c in ds.dims:
            return c
    return None


def subset_one_dataset(dataset_id, variables, start_date, end_date, output_filename):
    username = os.environ.get("CMEMS_USERNAME", "")
    password = os.environ.get("CMEMS_PASSWORD", "")

    if not username or not password:
        raise RuntimeError("CMEMS_USERNAME / CMEMS_PASSWORD が設定されていません。")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RAW_DIR / output_filename
    if output_path.exists():
        output_path.unlink()

    print("Downloading dataset:", dataset_id)
    print("Variables:", variables)

    kwargs = dict(
        dataset_id=dataset_id,
        variables=variables,
        minimum_longitude=LON_MIN,
        maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN,
        maximum_latitude=LAT_MAX,
        minimum_depth=0,
        maximum_depth=1,
        start_datetime=str(start_date),
        end_datetime=str(end_date),
        output_directory=str(RAW_DIR),
        output_filename=output_filename,
        file_format="netcdf",
        username=username,
        password=password,
        disable_progress_bar=True,
    )

    try:
        copernicusmarine.subset(**kwargs, overwrite=True)
    except TypeError:
        copernicusmarine.subset(**kwargs)

    if output_path.exists():
        return output_path

    candidates = list(RAW_DIR.glob(output_filename + "*"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"Downloaded file not found: {output_path}")


def subset_with_fallback(dataset_ids, variables, start_date, end_date, output_filename):
    errors = []

    for dataset_id in dataset_ids:
        try:
            return subset_one_dataset(
                dataset_id=dataset_id,
                variables=variables,
                start_date=start_date,
                end_date=end_date,
                output_filename=output_filename,
            )
        except Exception as e:
            print("Failed:", dataset_id)
            print(e)
            errors.append((dataset_id, str(e)))

    raise RuntimeError(f"All candidate datasets failed: {errors}")


def get_surface_da(ds, var, target_date):
    time_name = find_coord_name(ds, ["time"])

    da = ds[var].sel(
        {time_name: np.datetime64(str(target_date))},
        method="nearest"
    )

    depth_name = get_depth_name(ds)

    if depth_name is not None and depth_name in da.dims:
        da = da.isel({depth_name: 0})

    return da


def value_at_point(da, lat, lon):
    ds = da.to_dataset(name="tmp")

    lat_name = find_coord_name(ds, ["latitude", "lat"])
    lon_name = find_coord_name(ds, ["longitude", "lon"])

    try:
        val = da.interp(
            {
                lat_name: float(lat),
                lon_name: float(lon),
            }
        ).values
    except Exception:
        val = da.sel(
            {
                lat_name: float(lat),
                lon_name: float(lon),
            },
            method="nearest"
        ).values

    arr = np.asarray(val).astype(float)

    if arr.size == 0:
        return np.nan

    return float(np.nanmean(arr))


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return r * c


def current_core_points(speed_da):
    ds = speed_da.to_dataset(name="speed")

    lat_name = find_coord_name(ds, ["latitude", "lat"])
    lon_name = find_coord_name(ds, ["longitude", "lon"])

    speed = np.asarray(speed_da.values).astype(float)
    speed = np.squeeze(speed)

    lats = np.asarray(ds[lat_name].values).astype(float)
    lons = np.asarray(ds[lon_name].values).astype(float)

    if speed.ndim != 2:
        raise ValueError(f"speed_da is not 2D after squeeze. shape={speed.shape}")

    valid = np.isfinite(speed)

    if valid.sum() == 0:
        return np.array([]), np.array([]), np.nan

    threshold = np.nanquantile(speed[valid], 0.90)

    if not np.isfinite(threshold):
        return np.array([]), np.array([]), np.nan

    yy, xx = np.where((speed >= threshold) & np.isfinite(speed))

    core_lats = lats[yy]
    core_lons = lons[xx]

    return core_lats, core_lons, float(threshold)


def distance_to_current_core(lat, lon, core_lats, core_lons):
    if len(core_lats) == 0:
        return np.nan

    d = haversine_km(
        float(lat),
        float(lon),
        core_lats,
        core_lons,
    )

    return float(np.nanmin(d))


def main():
    if not FORECAST_ENV_PATH.exists():
        raise FileNotFoundError(f"Missing: {FORECAST_ENV_PATH}")

    base_date = jst_today()
    max_date = base_date + timedelta(days=max(FORECAST_DAYS_REAL))

    forecast = pd.read_csv(FORECAST_ENV_PATH)
    forecast["NetID"] = forecast["NetID"].astype(str)
    forecast["offset_days"] = pd.to_numeric(forecast["offset_days"], errors="coerce").astype(int)

    target_rows = forecast["offset_days"].isin(FORECAST_DAYS_REAL)

    if target_rows.sum() == 0:
        raise ValueError("forecast_env.csv に 0/3/7/10日の行がありません。")

    temp_path = subset_with_fallback(
        TEMP_DATASETS,
        ["thetao"],
        base_date,
        max_date,
        "copernicus_thetao_surface_forecast.nc",
    )

    cur_path = subset_with_fallback(
        CUR_DATASETS,
        ["uo", "vo"],
        base_date,
        max_date,
        "copernicus_current_surface_forecast.nc",
    )

    wcur_path = subset_with_fallback(
        WCUR_DATASETS,
        ["wo"],
        base_date,
        max_date,
        "copernicus_vertical_current_surface_forecast.nc",
    )

    print("Open datasets")
    ds_temp = xr.open_dataset(temp_path)
    ds_cur = xr.open_dataset(cur_path)
    ds_wcur = xr.open_dataset(wcur_path)

    updated_rows = []

    for idx, row in forecast.iterrows():
        offset = int(row["offset_days"])

        if offset not in FORECAST_DAYS_REAL:
            row["forecast_ocean_mode"] = "seasonal_climatology_fallback"
            updated_rows.append(row)
            continue

        target_date = base_date + timedelta(days=offset)

        thetao_da = get_surface_da(ds_temp, "thetao", target_date)
        uo_da = get_surface_da(ds_cur, "uo", target_date)
        vo_da = get_surface_da(ds_cur, "vo", target_date)
        wo_da = get_surface_da(ds_wcur, "wo", target_date)

        speed_da = np.sqrt(uo_da ** 2 + vo_da ** 2)

        core_lats, core_lons, core_threshold = current_core_points(speed_da)

        lat = float(row["Latitude"])
        lon = float(row["Longitude"])

        sst = value_at_point(thetao_da, lat, lon)
        uo = value_at_point(uo_da, lat, lon)
        vo = value_at_point(vo_da, lat, lon)
        wo = value_at_point(wo_da, lat, lon)

        if np.isfinite(uo) and np.isfinite(vo):
            current_speed = float(np.sqrt(uo ** 2 + vo ** 2))
        else:
            current_speed = np.nan

        kuro_dist = distance_to_current_core(lat, lon, core_lats, core_lons)

        row["SST"] = sst
        row["SST_source"] = "copernicus_forecast_thetao_surface"

        row["uo_forecast"] = uo
        row["vo_forecast"] = vo
        row["current_speed_forecast"] = current_speed
        row["wo_forecast"] = wo

        row["kuroshio_dist_km"] = kuro_dist
        row["kuroshio_source"] = "copernicus_forecast_surface_current_core_proxy"
        row["kuroshio_core_speed_threshold"] = core_threshold

        row["forecast_ocean_mode"] = "real_ocean_forecast"
        row["ocean_forecast_source"] = "Copernicus Marine GLOBAL_ANALYSISFORECAST_PHY_001_024"

        updated_rows.append(row)

    out = pd.DataFrame(updated_rows)

    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("Saved:", OUT_PATH)

    cols = [
        "offset_days",
        "target_date",
        "net_name",
        "NetID",
        "SST",
        "SST_source",
        "current_speed_forecast",
        "wo_forecast",
        "kuroshio_dist_km",
        "forecast_ocean_mode",
    ]

    cols = [c for c in cols if c in out.columns]

    print(out[cols].head(60))


if __name__ == "__main__":
    main()
