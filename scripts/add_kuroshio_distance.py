from pathlib import Path
import numpy as np
import pandas as pd


APP_ROOT = Path(__file__).resolve().parents[1]

NET_PATH = APP_ROOT / "data" / "net_master.csv"
AXIS_PATH = APP_ROOT / "data" / "kuroshio_axis.csv"
OUT_PATH = APP_ROOT / "data" / "net_master.csv"


def pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c

    if required:
        raise ValueError(f"Missing column. Tried: {candidates}")

    return None


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


def main():
    if not NET_PATH.exists():
        raise FileNotFoundError(f"Missing: {NET_PATH}")

    if not AXIS_PATH.exists():
        raise FileNotFoundError(f"Missing: {AXIS_PATH}")

    net = pd.read_csv(NET_PATH)
    axis = pd.read_csv(AXIS_PATH)

    net["NetID"] = net["NetID"].astype(str)

    net_lat_col = pick_col(net, ["Latitude", "latitude", "lat", "Lat"])
    net_lon_col = pick_col(net, ["Longitude", "longitude", "lon", "Lon"])

    axis_lat_col = pick_col(axis, ["axis_lat", "Latitude", "latitude", "lat", "Lat"])
    axis_lon_col = pick_col(axis, ["axis_lon", "Longitude", "longitude", "lon", "Lon"])

    axis[axis_lat_col] = pd.to_numeric(axis[axis_lat_col], errors="coerce")
    axis[axis_lon_col] = pd.to_numeric(axis[axis_lon_col], errors="coerce")

    axis = axis.dropna(subset=[axis_lat_col, axis_lon_col]).copy()

    # 高知周辺に絞って高速化
    axis = axis[
        (axis[axis_lon_col] >= 128) &
        (axis[axis_lon_col] <= 138) &
        (axis[axis_lat_col] >= 28) &
        (axis[axis_lat_col] <= 36)
    ].copy()

    if len(axis) == 0:
        raise ValueError("高知周辺の黒潮軸点がありません。kuroshio_axis.csv を確認してください。")

    print("Axis points used:", len(axis))

    axis_lats = axis[axis_lat_col].astype(float).values
    axis_lons = axis[axis_lon_col].astype(float).values

    dists = []

    for _, row in net.iterrows():
        lat = float(row[net_lat_col])
        lon = float(row[net_lon_col])

        d = haversine_km(
            lat,
            lon,
            axis_lats,
            axis_lons,
        )

        dists.append(float(np.nanmin(d)))

    net["kuroshio_dist_km"] = dists

    if "net_name" not in net.columns:
        net["net_name"] = "NetID " + net["NetID"].astype(str)

    net.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("Saved:", OUT_PATH)

    cols = ["net_name", "NetID", "Latitude", "Longitude", "kuroshio_dist_km"]
    cols = [c for c in cols if c in net.columns]

    print(net[cols].sort_values("kuroshio_dist_km"))


if __name__ == "__main__":
    main()
