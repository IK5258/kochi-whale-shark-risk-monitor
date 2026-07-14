from pathlib import Path
import os

import numpy as np
import pandas as pd


APP_ROOT = Path(
    os.environ.get(
        "APP_ROOT",
        Path(__file__).resolve().parents[1],
    )
)

MASTER_PATH = (
    APP_ROOT
    / "data"
    / "WhaleShark_env_master.csv"
)

NAME_PATH = (
    APP_ROOT
    / "data"
    / "net_name_master.csv"
)

OUT_PATH = (
    APP_ROOT
    / "data"
    / "net_master.csv"
)


def pick_col(df, candidates, required=True):
    for column in candidates:
        if column in df.columns:
            return column

    if required:
        raise ValueError(
            "Missing required column. "
            f"Tried: {candidates}"
        )

    return None


def normalize_net_id(series):
    return (
        series
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )


def main():
    if not MASTER_PATH.exists():
        raise FileNotFoundError(
            f"Master CSV not found: {MASTER_PATH}"
        )

    if not NAME_PATH.exists():
        raise FileNotFoundError(
            f"Net name master not found: {NAME_PATH}"
        )

    df = pd.read_csv(
        MASTER_PATH,
        low_memory=False,
    )

    net_col = pick_col(
        df,
        ["NetID", "net_id", "net", "station", "Station"],
    )

    lat_col = pick_col(
        df,
        ["Latitude", "latitude", "lat", "Lat"],
    )

    lon_col = pick_col(
        df,
        ["Longitude", "longitude", "lon", "Lon"],
    )

    depth_col = pick_col(
        df,
        [
            "depth_m",
            "Depth_m",
            "depth",
            "Depth",
            "bathymetry_m",
        ],
    )

    presence_col = pick_col(
        df,
        [
            "presence",
            "Presence",
            "pa",
            "PA",
            "occurrence",
        ],
        required=False,
    )

    kuro_col = pick_col(
        df,
        [
            "kuroshio_dist_km",
            "Kuroshio_dist_km",
            "dist_kuroshio_km",
            "kuroshio_distance_km",
            "Kuroshio_distance_km",
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
            "cold_sst_anomaly_z",
        ],
        required=False,
    )

    ret_col = pick_col(
        df,
        [
            "retention_proxy_convergence",
            "retention_proxy_convergence_rank",
            "current_convergence",
            "current_convergence_z",
        ],
        required=False,
    )

    use_cols = [
        net_col,
        lat_col,
        lon_col,
        depth_col,
    ]

    for column in [
        presence_col,
        kuro_col,
        up_col,
        ret_col,
    ]:
        if column:
            use_cols.append(column)

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

    dat["NetID"] = normalize_net_id(
        dat["NetID"]
    )

    dat["Latitude"] = pd.to_numeric(
        dat["Latitude"],
        errors="coerce",
    )

    dat["Longitude"] = pd.to_numeric(
        dat["Longitude"],
        errors="coerce",
    )

    dat["depth_m"] = (
        pd.to_numeric(
            dat["depth_m"],
            errors="coerce",
        )
        .abs()
    )

    if "presence" in dat.columns:
        dat["presence"] = (
            pd.to_numeric(
                dat["presence"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )
    else:
        dat["presence"] = 1

    for column in [
        "kuroshio_dist_km",
        "upwelling_proxy",
        "retention_proxy",
    ]:
        if column in dat.columns:
            dat[column] = pd.to_numeric(
                dat[column],
                errors="coerce",
            )

    agg_dict = {
        "Latitude": "mean",
        "Longitude": "mean",
        "depth_m": "median",
        "presence": "sum",
    }

    for column in [
        "kuroshio_dist_km",
        "upwelling_proxy",
        "retention_proxy",
    ]:
        if column in dat.columns:
            agg_dict[column] = "median"

    net = (
        dat
        .dropna(
            subset=[
                "NetID",
                "Latitude",
                "Longitude",
                "depth_m",
            ]
        )
        .groupby(
            "NetID",
            as_index=False,
        )
        .agg(agg_dict)
        .rename(
            columns={
                "presence": "n_presence",
            }
        )
    )

    for column in [
        "kuroshio_dist_km",
        "upwelling_proxy",
        "retention_proxy",
    ]:
        if column not in net.columns:
            net[column] = np.nan

    # 正式な定置網名を読み込む
    names = pd.read_csv(
        NAME_PATH,
        encoding="utf-8-sig",
    )

    names.columns = [
        str(column)
        .replace("\ufeff", "")
        .strip()
        for column in names.columns
    ]

    required_name_columns = {
        "NetID",
        "net_name",
    }

    missing_columns = (
        required_name_columns
        - set(names.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing columns in net_name_master.csv: "
            + ", ".join(sorted(missing_columns))
        )

    names["NetID"] = normalize_net_id(
        names["NetID"]
    )

    names["net_name"] = (
        names["net_name"]
        .astype(str)
        .str.strip()
    )

    names = (
        names[
            [
                "NetID",
                "net_name",
            ]
        ]
        .drop_duplicates(
            subset=["NetID"]
        )
    )

    # NetIDをキーに正式名称を結合
    net = net.merge(
        names,
        on="NetID",
        how="left",
        validate="one_to_one",
    )

    missing_name = (
        net["net_name"].isna()
        | net["net_name"].eq("")
        | net["net_name"].str.lower().eq("nan")
    )

    if missing_name.any():
        missing_ids = (
            net.loc[
                missing_name,
                "NetID",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Official net names are missing for NetID: "
            + ", ".join(missing_ids)
        )

    net["net_label"] = (
        net["net_name"]
        + "（NetID "
        + net["NetID"]
        + "）"
    )

    # NetIDを数値順に並べる
    net["_sort_id"] = pd.to_numeric(
        net["NetID"],
        errors="coerce",
    )

    net = (
        net
        .sort_values(
            ["_sort_id", "NetID"]
        )
        .drop(columns="_sort_id")
        .reset_index(drop=True)
    )

    OUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    net.to_csv(
        OUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Saved: {OUT_PATH}")

    print(
        net[
            [
                "NetID",
                "net_name",
                "net_label",
            ]
        ]
    )

    print(
        "Official set-net names loaded:",
        len(net),
    )


if __name__ == "__main__":
    main()
