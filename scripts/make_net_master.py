from pathlib import Path
import os
import pandas as pd
import numpy as np


APP_ROOT = Path(os.environ.get("APP_ROOT", Path(__file__).resolve().parents[1]))

MASTER_PATH = APP_ROOT / "data" / "WhaleShark_env_master.csv"
OUT_PATH = APP_ROOT / "data" / "net_master.csv"


def pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}")
    return None


def main():
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Master CSV not found: {MASTER_PATH}")

    df = pd.read_csv(MASTER_PATH)

    net_col = pick_col(df, ["NetID", "net_id", "net", "station", "Station"], required=True)
    lat_col = pick_col(df, ["Latitude", "latitude", "lat", "Lat"], required=True)
    lon_col = pick_col(df, ["Longitude", "longitude", "lon", "Lon"], required=True)
    depth_col = pick_col(df, ["depth_m", "Depth_m", "depth", "Depth", "bathymetry_m"], required=True)

    presence_col = pick_col(
        df,
        ["presence", "Presence", "pa", "PA", "occurrence"],
        required=False
    )

    kuro_col = pick_col(
        df,
        [
            "kuroshio_dist_km",
            "Kuroshio_dist_km",
            "dist_kuroshio_km",
            "kuroshio_distance_km",
            "Kuroshio_distance_km"
        ],
        required=False,
    )

    up_col = pick_col(
        df,
        [
            "upwelling_proxy",
            "Upwelling_proxy",
            "upwelling",
            "upwelling_score",
            "upwelling_proxy_divergence",
            "upwelling_proxy_divergence_rank",
            "current_divergence",
            "current_divergence_z",
            "cold_sst_anomaly",
            "cold_sst_anomaly_z"
        ],
        required=False,
    )

    ret_col = pick_col(
        df,
        [
            "retention_proxy_convergence",
            "retention_proxy_convergence_rank",
            "current_convergence",
            "current_convergence_z"
        ],
        required=False,
    )

    use_cols = [net_col, lat_col, lon_col, depth_col]

    if presence_col:
        use_cols.append(presence_col)
    if kuro_col:
        use_cols.append(kuro_col)
    if up_col:
        use_cols.append(up_col)
    if ret_col:
        use_cols.append(ret_col)

    dat = df[use_cols].copy()

    rename = {
        net_col: "NetID",
        lat_col: "Latitude",
        lon_col: "Longitude",
        depth_col: "depth_m",
    }

    if presence_col:
        rename[presence_col] = "presence"
    if kuro_col:
        rename[kuro_col] = "kuroshio_dist_km"
    if up_col:
        rename[up_col] = "upwelling_proxy"
    if ret_col:
        rename[ret_col] = "retention_proxy"

    dat = dat.rename(columns=rename)

    dat["NetID"] = dat["NetID"].astype(str)
    dat["Latitude"] = pd.to_numeric(dat["Latitude"], errors="coerce")
    dat["Longitude"] = pd.to_numeric(dat["Longitude"], errors="coerce")
    dat["depth_m"] = pd.to_numeric(dat["depth_m"], errors="coerce").abs()

    if "presence" in dat.columns:
        dat["presence"] = pd.to_numeric(dat["presence"], errors="coerce").fillna(0).astype(int)
    else:
        dat["presence"] = 1

    if "kuroshio_dist_km" in dat.columns:
        dat["kuroshio_dist_km"] = pd.to_numeric(dat["kuroshio_dist_km"], errors="coerce")

    if "upwelling_proxy" in dat.columns:
        dat["upwelling_proxy"] = pd.to_numeric(dat["upwelling_proxy"], errors="coerce")

    if "retention_proxy" in dat.columns:
        dat["retention_proxy"] = pd.to_numeric(dat["retention_proxy"], errors="coerce")

    agg_dict = {
        "Latitude": "mean",
        "Longitude": "mean",
        "depth_m": "median",
        "presence": "sum",
    }

    if "kuroshio_dist_km" in dat.columns:
        agg_dict["kuroshio_dist_km"] = "median"

    if "upwelling_proxy" in dat.columns:
        agg_dict["upwelling_proxy"] = "median"

    if "retention_proxy" in dat.columns:
        agg_dict["retention_proxy"] = "median"

    net = (
        dat
        .dropna(subset=["NetID", "Latitude", "Longitude", "depth_m"])
        .groupby("NetID", as_index=False)
        .agg(agg_dict)
        .rename(columns={"presence": "n_presence"})
        .sort_values("NetID")
    )

    if "kuroshio_dist_km" not in net.columns:
        net["kuroshio_dist_km"] = np.nan

    if "upwelling_proxy" not in net.columns:
        net["upwelling_proxy"] = np.nan

    if "retention_proxy" not in net.columns:
        net["retention_proxy"] = np.nan

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    net.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved: {OUT_PATH}")
    print(net.head())
    print("\ncolumns:")
    print(net.columns.tolist())


if __name__ == "__main__":
    main()
