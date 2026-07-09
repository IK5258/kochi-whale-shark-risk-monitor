from pathlib import Path
import numpy as np
import pandas as pd


APP_ROOT = Path(__file__).resolve().parents[1]

NET_PATH = APP_ROOT / "data" / "net_master.csv"
CURRENT_PATH = APP_ROOT / "outputs" / "current_env.csv"
RISK_PATH = APP_ROOT / "outputs" / "latest_risk.csv"


def kuroshio_context_from_distance(dist_km):
    return np.exp(-0.5 * ((dist_km - 265.6) / 180.0) ** 2)


def classify_rank(p):
    if pd.isna(p):
        return "Unknown"
    if p >= 0.90:
        return "Very high"
    if p >= 0.75:
        return "High"
    if p >= 0.50:
        return "Moderate"
    return "Low"


def apply_to_file(path, net):
    if not path.exists():
        print("Skip, not found:", path)
        return

    df = pd.read_csv(path)
    df["NetID"] = df["NetID"].astype(str)

    drop_cols = [
        "kuroshio_dist_km",
        "kuroshio_context",
    ]

    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    use_cols = ["NetID", "kuroshio_dist_km"]

    if "net_name" in net.columns and "net_name" not in df.columns:
        use_cols.append("net_name")

    df = df.merge(net[use_cols], on="NetID", how="left")

    df["kuroshio_context"] = kuroshio_context_from_distance(
        pd.to_numeric(df["kuroshio_dist_km"], errors="coerce")
    )

    if path.name == "latest_risk.csv":
        core = pd.to_numeric(df["core_risk"], errors="coerce")

        if "upwelling_context" in df.columns:
            upwelling = pd.to_numeric(df["upwelling_context"], errors="coerce").fillna(0.5)
        else:
            upwelling = 0.5

        kuro = pd.to_numeric(df["kuroshio_context"], errors="coerce").fillna(0.5)

        df["integrated_risk"] = 0.75 * core + 0.15 * kuro + 0.10 * upwelling
        df["integrated_formula"] = "0.75 core + 0.15 kuroshio + 0.10 upwelling"

        df["integrated_percentile_today"] = df["integrated_risk"].rank(pct=True)
        df["integrated_risk_class"] = df["integrated_percentile_today"].apply(classify_rank)

        df["note"] = (
            "Integrated risk is experimental. "
            "Core GAM risk should be treated as the main risk index. "
            "Kuroshio context is included."
        )

    df.to_csv(path, index=False, encoding="utf-8-sig")
    print("Updated:", path)


def main():
    net = pd.read_csv(NET_PATH)
    net["NetID"] = net["NetID"].astype(str)

    if "kuroshio_dist_km" not in net.columns:
        raise ValueError("net_master.csv に kuroshio_dist_km がありません。")

    apply_to_file(CURRENT_PATH, net)
    apply_to_file(RISK_PATH, net)

    risk = pd.read_csv(RISK_PATH)

    cols = [
        "net_name",
        "NetID",
        "core_risk",
        "kuroshio_dist_km",
        "kuroshio_context",
        "upwelling_context",
        "integrated_risk",
        "integrated_formula",
    ]

    cols = [c for c in cols if c in risk.columns]

    print(risk[cols].sort_values("integrated_risk", ascending=False))


if __name__ == "__main__":
    main()
