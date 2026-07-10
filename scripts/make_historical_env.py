
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr


APP_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = APP_ROOT / "data"
OUTPUT_DIR = APP_ROOT / "outputs"

WORK_DIR = APP_ROOT.parent
SST_DIR = WORK_DIR / "oisst_regional_ncss"

NET_PATH = DATA_DIR / "net_master.csv"
NET_NAME_PATH = DATA_DIR / "net_name_master.csv"

OUT_PATH = OUTPUT_DIR / "historical_env_daily.csv"


def find_coord(da, candidates):
    for c in da.coords:
        if c.lower() in candidates:
            return c
    for d in da.dims:
        if d.lower() in candidates:
            return d
    raise ValueError(f"coordinate not found: coords={list(da.coords)}, dims={list(da.dims)}")


def find_sst_var(ds):
    for v in ds.data_vars:
        vl = v.lower()
        if vl in ["sst", "sea_surface_temperature"]:
            return v
        if "sst" in vl:
            return v
    return list(ds.data_vars)[0]


def pick_col(df, candidates, required=True):
    lower = {str(c).lower(): c for c in df.columns}

    for c in candidates:
        if c in df.columns:
            return c

    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]

    if required:
        raise ValueError(f"column not found: {candidates}. columns={df.columns.tolist()}")

    return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    nets = pd.read_csv(NET_PATH)

    netid_col = pick_col(nets, ["NetID", "net_id", "net"])
    lat_col = pick_col(nets, ["Latitude", "latitude", "lat", "Lat"])
    lon_col = pick_col(nets, ["Longitude", "longitude", "lon", "Lon"])
    depth_col = pick_col(nets, ["depth_m", "Depth_m", "depth", "Depth", "bathymetry_m"])

    nets = nets.rename(columns={
        netid_col: "NetID",
        lat_col: "Latitude",
        lon_col: "Longitude",
        depth_col: "depth_m",
    })

    nets["NetID"] = nets["NetID"].astype(str)
    nets["Latitude"] = pd.to_numeric(nets["Latitude"], errors="coerce")
    nets["Longitude"] = pd.to_numeric(nets["Longitude"], errors="coerce")
    nets["depth_m"] = abs(pd.to_numeric(nets["depth_m"], errors="coerce"))

    if NET_NAME_PATH.exists():
        names = pd.read_csv(NET_NAME_PATH)
        name_netid_col = pick_col(names, ["NetID", "net_id", "net"])
        name_col = pick_col(names, ["net_name", "Name", "name", "net_label"], required=False)

        names = names.rename(columns={name_netid_col: "NetID"})
        names["NetID"] = names["NetID"].astype(str)

        if name_col is not None:
            names = names.rename(columns={name_col: "net_name"})
            nets = nets.merge(names[["NetID", "net_name"]], on="NetID", how="left")

    if "net_name" not in nets.columns:
        nets["net_name"] = nets["NetID"]

    sst_files = sorted(SST_DIR.glob("OISST_Kochi_*.nc"))

    rows = []

    for f in sst_files:
        year_text = f.stem.split("_")[-1]

        try:
            year = int(year_text)
        except Exception:
            continue

        # 過去10年くらい。2016年以降だけ使用。
        if year < 2016:
            continue

        print("reading:", f)

        ds = xr.open_dataset(f)
        sst_name = find_sst_var(ds)
        sst = ds[sst_name]

        time_name = find_coord(sst, ["time"])
        lat_name = find_coord(sst, ["lat", "latitude", "y"])
        lon_name = find_coord(sst, ["lon", "longitude", "x"])

        for _, net in nets.iterrows():
            sst_net = sst.sel(
                {
                    lat_name: float(net["Latitude"]),
                    lon_name: float(net["Longitude"]),
                },
                method="nearest"
            )

            tmp = pd.DataFrame({
                "date": pd.to_datetime(sst_net[time_name].values),
                "SST": np.asarray(sst_net.values).reshape(-1),
            })

            tmp["Date"] = pd.to_datetime(tmp["date"]).dt.date
            tmp["target_date"] = tmp["Date"]
            tmp["Year"] = pd.to_datetime(tmp["Date"]).dt.year
            tmp["Month"] = pd.to_datetime(tmp["Date"]).dt.month
            tmp["Jday"] = pd.to_datetime(tmp["Date"]).dt.dayofyear

            tmp["NetID"] = net["NetID"]
            tmp["Latitude"] = net["Latitude"]
            tmp["Longitude"] = net["Longitude"]
            tmp["depth_m"] = net["depth_m"]
            tmp["net_name"] = net["net_name"]
            tmp["SST_source"] = f.name

            rows.append(tmp)

        ds.close()

    hist = pd.concat(rows, ignore_index=True)

    hist["NetID"] = hist["NetID"].astype(str)
    hist["net_label"] = hist["net_name"].astype(str) + " / NetID " + hist["NetID"].astype(str)
    hist["historical_mode"] = "daily_oisst"
    hist["note"] = "Historical risk is based on OISST, Jday, and GEBCO depth. Kuroshio and upwelling contexts are not included."

    hist = hist.replace([np.inf, -np.inf], np.nan)
    hist = hist.dropna(subset=["Date", "Jday", "SST", "depth_m", "Latitude", "Longitude"])

    hist.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("saved:", OUT_PATH)
    print(hist.shape)
    print(hist.head())


if __name__ == "__main__":
    main()
