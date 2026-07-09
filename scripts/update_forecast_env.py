from pathlib import Path
from datetime import datetime, timedelta, timezone
import os
import numpy as np
import pandas as pd


APP_ROOT = Path(os.environ.get("APP_ROOT", Path(__file__).resolve().parents[1]))

MASTER_PATH = APP_ROOT / "data" / "WhaleShark_env_master.csv"
NET_PATH = APP_ROOT / "data" / "net_master.csv"

OUT_PATH = APP_ROOT / "outputs" / "forecast_env.csv"
CURRENT_OUT_PATH = APP_ROOT / "outputs" / "current_env.csv"

FORECAST_DAYS = [0, 3, 7, 10, 14, 30]


def jday_from_date(d):
    return int(d.strftime("%j"))


def circular_jday_distance(a, b):
    diff = abs(a - b)
    return min(diff, 366 - diff)


def estimate_sst(master, net_id, target_jday, window=10):
    df = master.copy()

    if "NetID" in df.columns:
        df["NetID"] = df["NetID"].astype(str)
        cand = df[df["NetID"] == str(net_id)].copy()
    else:
        cand = df.copy()

    if len(cand) == 0:
        cand = df.copy()

    if "Jday" not in cand.columns or "SST" not in cand.columns:
        return np.nan

    cand["Jday"] = pd.to_numeric(cand["Jday"], errors="coerce")
    cand["SST"] = pd.to_numeric(cand["SST"], errors="coerce")

    cand = cand.dropna(subset=["Jday", "SST"]).copy()

    if len(cand) == 0:
        return np.nan

    cand["jday_dist"] = cand["Jday"].apply(lambda x: circular_jday_distance(int(x), int(target_jday)))

    near = cand[cand["jday_dist"] <= window].copy()

    if len(near) >= 5:
        return float(near["SST"].median())

    near = cand[cand["jday_dist"] <= 20].copy()

    if len(near) >= 5:
        return float(near["SST"].median())

    return float(cand["SST"].median())


def main():
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Missing: {MASTER_PATH}")

    if not NET_PATH.exists():
        raise FileNotFoundError(f"Missing: {NET_PATH}")

    master = pd.read_csv(MASTER_PATH)
    net = pd.read_csv(NET_PATH)

    net["NetID"] = net["NetID"].astype(str)

    run_date_env = os.environ.get("RUN_DATE", "")

    if run_date_env:
        base_date = datetime.strptime(run_date_env, "%Y-%m-%d").date()
    else:
        base_date = datetime.now(timezone(timedelta(hours=9))).date()

    rows = []

    for offset in FORECAST_DAYS:
        target_date = base_date + timedelta(days=offset)
        target_jday = jday_from_date(target_date)

        for _, r in net.iterrows():
            net_id = str(r["NetID"])

            sst = estimate_sst(master, net_id, target_jday, window=10)

            row = {
                "forecast_base_date": str(base_date),
                "target_date": str(target_date),
                "date": str(target_date),
                "offset_days": offset,
                "forecast_label": "Today" if offset == 0 else f"{offset} days later",
                "Jday": target_jday,
                "NetID": net_id,
                "Latitude": r.get("Latitude", np.nan),
                "Longitude": r.get("Longitude", np.nan),
                "depth_m": r.get("depth_m", np.nan),
                "SST": sst,
                "SST_source": "historical_climatology_by_NetID_and_Jday_window",
            }

            for c in [
                "net_name",
                "net_label",
                "n_presence",
                "kuroshio_dist_km",
                "upwelling_proxy",
                "retention_proxy",
            ]:
                if c in net.columns:
                    row[c] = r.get(c, np.nan)

            rows.append(row)

    out = pd.DataFrame(rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    current = out[out["offset_days"] == 0].copy()
    current.to_csv(CURRENT_OUT_PATH, index=False, encoding="utf-8-sig")

    print("Saved:", OUT_PATH)
    print("Saved:", CURRENT_OUT_PATH)
    print(out[["target_date", "offset_days", "net_name", "NetID", "Jday", "SST", "depth_m", "kuroshio_dist_km"]].head(20))


if __name__ == "__main__":
    main()
