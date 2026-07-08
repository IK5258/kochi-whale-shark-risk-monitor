from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import os
import pandas as pd
import numpy as np


APP_ROOT = Path(os.environ.get("APP_ROOT", Path(__file__).resolve().parents[1]))

MASTER_PATH = APP_ROOT / "data" / "WhaleShark_env_master.csv"
NET_PATH = APP_ROOT / "data" / "net_master.csv"
OUT_PATH = APP_ROOT / "outputs" / "current_env.csv"
LOG_PATH = APP_ROOT / "outputs" / "update_log.csv"


def pick_col(df, candidates, required=False):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}")
    return None


def circular_day_distance(a, b, year_len=366):
    d = abs(a - b)
    return min(d, year_len - d)


def estimate_sst_from_master(master, net_id, jday):
    if "SST" not in master.columns or "Jday" not in master.columns:
        return np.nan

    sub = master.copy()
    sub["Jday"] = pd.to_numeric(sub["Jday"], errors="coerce")
    sub["SST"] = pd.to_numeric(sub["SST"], errors="coerce")

    sub = sub.dropna(subset=["Jday", "SST"])

    if "NetID" in sub.columns:
        sub_net = sub[sub["NetID"].astype(str) == str(net_id)]
    else:
        sub_net = pd.DataFrame()

    for window in [7, 14, 30, 60, 120, 183]:
        if len(sub_net) > 0:
            tmp = sub_net[sub_net["Jday"].apply(lambda x: circular_day_distance(x, jday) <= window)]
            if len(tmp) >= 3:
                return float(tmp["SST"].median())

        tmp = sub[sub["Jday"].apply(lambda x: circular_day_distance(x, jday) <= window)]
        if len(tmp) >= 10:
            return float(tmp["SST"].median())

    return float(sub["SST"].median())


def main():
    if not NET_PATH.exists():
        raise FileNotFoundError(f"net_master.csv not found. Run make_net_master.py first: {NET_PATH}")

    net = pd.read_csv(NET_PATH)
    master = pd.read_csv(MASTER_PATH) if MASTER_PATH.exists() else pd.DataFrame()

    if "NetID" in master.columns:
        master["NetID"] = master["NetID"].astype(str)

    run_date_env = os.environ.get("RUN_DATE", "")
    if run_date_env:
        today = datetime.strptime(run_date_env, "%Y-%m-%d").replace(tzinfo=ZoneInfo("Asia/Tokyo"))
    else:
        today = datetime.now(ZoneInfo("Asia/Tokyo"))

    run_date = today.strftime("%Y-%m-%d")
    jday = int(today.strftime("%j"))

    current = net.copy()
    current["NetID"] = current["NetID"].astype(str)
    current["date"] = run_date
    current["Jday"] = jday

    if "SST" not in current.columns:
        current["SST"] = current["NetID"].apply(lambda x: estimate_sst_from_master(master, x, jday))

    if current["SST"].isna().all():
        current["SST"] = 28.0

    current["SST_source"] = "master_climatology_v0"

    for c in ["kuroshio_dist_km", "upwelling_proxy"]:
        if c not in current.columns:
            current[c] = np.nan

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    current.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    log_row = pd.DataFrame([{
        "datetime_jst": datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M:%S"),
        "date": run_date,
        "Jday": jday,
        "n_nets": len(current),
        "sst_source": "master_climatology_v0",
        "status": "success",
    }])

    if LOG_PATH.exists():
        old = pd.read_csv(LOG_PATH)
        log = pd.concat([old, log_row], ignore_index=True)
    else:
        log = log_row

    log.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved: {OUT_PATH}")
    print(current.head())


if __name__ == "__main__":
    main()
