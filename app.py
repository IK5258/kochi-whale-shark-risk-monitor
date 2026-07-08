from pathlib import Path
import pandas as pd
import streamlit as st
import pydeck as pdk


APP_ROOT = Path(__file__).resolve().parent
RISK_PATH = APP_ROOT / "outputs" / "latest_risk.csv"
LOG_PATH = APP_ROOT / "outputs" / "update_log.csv"

st.set_page_config(
    page_title="Kochi Whale Shark Risk Monitor",
    page_icon="🦈",
    layout="wide",
)

st.title("Kochi Whale Shark Risk Monitor")
st.caption("高知県沿岸の定置網におけるジンベエザメ出現リスクモニター v0.1")


@st.cache_data
def load_risk():
    if not RISK_PATH.exists():
        return None
    return pd.read_csv(RISK_PATH)


@st.cache_data
def load_log():
    if not LOG_PATH.exists():
        return None
    return pd.read_csv(LOG_PATH)


risk = load_risk()
log = load_log()

if risk is None:
    st.error("latest_risk.csv が見つかりません。scripts/update_daily_env.py と scripts/predict_risk.R を実行してください。")
    st.stop()

required_cols = ["NetID", "Latitude", "Longitude", "core_risk", "integrated_risk"]

missing = [c for c in required_cols if c not in risk.columns]
if missing:
    st.error(f"latest_risk.csv に必要な列がありません: {missing}")
    st.stop()

risk["NetID"] = risk["NetID"].astype(str)

st.sidebar.header("表示設定")

risk_mode = st.sidebar.radio(
    "表示するリスク",
    ["Core GAM risk", "Integrated risk", "MaxEnt suitability"],
    index=0,
)

show_context = st.sidebar.checkbox("湧昇proxyを表示", value=True)

if "date" in risk.columns:
    latest_date = str(risk["date"].iloc[0])
else:
    latest_date = "Unknown"

if "Jday" in risk.columns:
    latest_jday = int(risk["Jday"].iloc[0])
else:
    latest_jday = None

st.sidebar.markdown("---")
st.sidebar.write(f"Date: {latest_date}")
if latest_jday is not None:
    st.sidebar.write(f"Jday: {latest_jday}")

if log is not None and len(log) > 0:
    st.sidebar.markdown("---")
    st.sidebar.write("Last update")
    st.sidebar.dataframe(log.tail(3), use_container_width=True)

if risk_mode == "Core GAM risk":
    value_col = "core_risk"
    class_col = "core_risk_class" if "core_risk_class" in risk.columns else None
elif risk_mode == "Integrated risk":
    value_col = "integrated_risk"
    class_col = "integrated_risk_class" if "integrated_risk_class" in risk.columns else None
else:
    value_col = "maxent_suitability" if "maxent_suitability" in risk.columns else "core_risk"
    class_col = None

risk[value_col] = pd.to_numeric(risk[value_col], errors="coerce")
risk = risk.dropna(subset=["Latitude", "Longitude", value_col])

top = risk.sort_values(value_col, ascending=False).iloc[0]

col1, col2, col3, col4 = st.columns(4)

col1.metric("Top NetID", top["NetID"])
col2.metric("Risk score", f"{top[value_col]:.3f}")

if "SST" in risk.columns:
    col3.metric("SST", f"{float(top['SST']):.1f} ℃")
else:
    col3.metric("SST", "NA")

if "depth_m" in risk.columns:
    col4.metric("Depth", f"{float(top['depth_m']):.0f} m")
else:
    col4.metric("Depth", "NA")

st.markdown("### Risk map")


def risk_color(x):
    if pd.isna(x):
        return [180, 180, 180]
    if x >= 0.75:
        return [220, 40, 40]
    elif x >= 0.50:
        return [240, 150, 40]
    elif x >= 0.25:
        return [240, 220, 80]
    else:
        return [60, 160, 90]


plot_df = risk.copy()
plot_df["plot_value"] = plot_df[value_col]

vmin = plot_df["plot_value"].min()
vmax = plot_df["plot_value"].max()

if vmax > vmin:
    plot_df["plot_scaled"] = (plot_df["plot_value"] - vmin) / (vmax - vmin)
else:
    plot_df["plot_scaled"] = 0.5

plot_df["radius"] = 2500 + plot_df["plot_scaled"] * 6000
plot_df["color"] = plot_df["plot_scaled"].apply(risk_color)

tooltip_html = "<b>NetID:</b> {NetID}<br>"
tooltip_html += f"<b>{value_col}:</b> " + "{" + value_col + "}<br>"

for c in [
    "core_risk",
    "maxent_suitability",
    "integrated_risk",
    "SST",
    "depth_m",
    "upwelling_proxy",
    "upwelling_context",
]:
    if c in plot_df.columns and c != value_col:
        tooltip_html += f"<b>{c}:</b> " + "{" + c + "}<br>"

layer = pdk.Layer(
    "ScatterplotLayer",
    data=plot_df,
    get_position="[Longitude, Latitude]",
    get_radius="radius",
    get_fill_color="color",
    pickable=True,
    opacity=0.75,
)

view_state = pdk.ViewState(
    latitude=float(plot_df["Latitude"].mean()),
    longitude=float(plot_df["Longitude"].mean()),
    zoom=7.5,
    pitch=0,
)

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"html": tooltip_html, "style": {"backgroundColor": "white", "color": "black"}},
)

st.pydeck_chart(deck, use_container_width=True)

st.markdown("### Risk ranking")

display_cols = [
    "NetID",
    "date",
    "Jday",
    "SST",
    "depth_m",
    "core_risk",
    "core_risk_class",
    "maxent_suitability",
    "upwelling_context",
    "integrated_risk",
    "integrated_risk_class",
    "integrated_formula",
]

display_cols = [c for c in display_cols if c in risk.columns]

rank_df = risk.sort_values(value_col, ascending=False)[display_cols].copy()

for c in rank_df.columns:
    if rank_df[c].dtype.kind in "fc":
        rank_df[c] = rank_df[c].round(3)

st.dataframe(rank_df, use_container_width=True, hide_index=True)

if show_context:
    st.markdown("### Ecological context")

    st.subheader("Upwelling proxy")

    if "upwelling_proxy" in risk.columns:
        tmp = risk[["NetID", "upwelling_proxy", "upwelling_context"]].copy()
        tmp = tmp.sort_values("upwelling_context", ascending=False)

        for c in tmp.columns:
            if tmp[c].dtype.kind in "fc":
                tmp[c] = tmp[c].round(3)

        st.dataframe(tmp, use_container_width=True, hide_index=True)
    else:
        st.info("upwelling_proxy がありません。")

st.markdown("---")
st.markdown("### Model note")

st.write(
    """
    Core risk は GAM による主リスクです。
    モデル式は `presence ~ s(Jday, bs="cc") + s(SST) + s(depth_m)` です。

    MaxEnt suitability は補助的な出現適性指標です。

    Integrated risk は実験的な総合スコアです。
    v0.1 では Core GAM risk を主軸に、湧昇proxyのみを補助的に加えています。

    Integrated risk = 0.85 × Core GAM risk + 0.15 × Upwelling context

    黒潮proxyは、過去解析では生態学的に関与している可能性が示されましたが、
    現在のアプリ v0.1 にはまだ実装していません。
    今後、黒潮軸距離データ `kuroshio_dist_km` を追加した v0.2 で実装予定です。

    `integrated_risk_class` は、その日の定置網間での相対ランクです。
    絶対的な危険度ではなく、当日どの定置網が相対的に高いかを見るための指標です。
    """
)
