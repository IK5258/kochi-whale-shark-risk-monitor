from pathlib import Path
import pandas as pd
import streamlit as st
import pydeck as pdk


APP_ROOT = Path(__file__).resolve().parent

FORECAST_PATH = APP_ROOT / "outputs" / "forecast_risk.csv"
RISK_PATH = APP_ROOT / "outputs" / "latest_risk.csv"
LOG_PATH = APP_ROOT / "outputs" / "update_log.csv"

st.set_page_config(
    page_title="Kochi Whale Shark Risk Monitor",
    page_icon="🦈",
    layout="wide",
)


def check_password():
    try:
        app_password = st.secrets["APP_PASSWORD"]
    except Exception:
        st.error("APP_PASSWORD が設定されていません。Streamlit Cloud の Secrets に設定してください。")
        st.stop()

    if "password_ok" not in st.session_state:
        st.session_state["password_ok"] = False

    if st.session_state["password_ok"]:
        return

    st.title("Kochi Whale Shark Risk Monitor")
    st.caption("このアプリの閲覧にはパスワードが必要です。")

    password = st.text_input("Password", type="password")

    if password:
        if password == app_password:
            st.session_state["password_ok"] = True
            st.rerun()
        else:
            st.error("パスワードが違います。")
            st.stop()
    else:
        st.stop()


check_password()

st.title("Kochi Whale Shark Risk Monitor")
st.caption("高知県沿岸の定置網におけるジンベエザメ出現リスクモニター v0.3")


@st.cache_data
def load_risk():
    if FORECAST_PATH.exists():
        return pd.read_csv(FORECAST_PATH), "forecast_risk.csv"

    if RISK_PATH.exists():
        return pd.read_csv(RISK_PATH), "latest_risk.csv"

    return None, None


@st.cache_data
def load_log():
    if not LOG_PATH.exists():
        return None
    return pd.read_csv(LOG_PATH)


risk_all, source_file = load_risk()
log = load_log()

if risk_all is None:
    st.error("risk csv が見つかりません。forecast_risk.csv または latest_risk.csv を確認してください。")
    st.stop()

risk_all["NetID"] = risk_all["NetID"].astype(str)

if "net_name" not in risk_all.columns:
    risk_all["net_name"] = "NetID " + risk_all["NetID"]

if "net_label" not in risk_all.columns:
    risk_all["net_label"] = risk_all["net_name"] + "（NetID " + risk_all["NetID"] + "）"

if "offset_days" not in risk_all.columns:
    risk_all["offset_days"] = 0

if "target_date" not in risk_all.columns:
    if "date" in risk_all.columns:
        risk_all["target_date"] = risk_all["date"]
    else:
        risk_all["target_date"] = "Unknown"

risk_all["offset_days"] = pd.to_numeric(risk_all["offset_days"], errors="coerce").fillna(0).astype(int)

available_offsets = sorted(risk_all["offset_days"].unique().tolist())

offset_label_map = {
    0: "今日",
    3: "3日後",
    7: "7日後",
    10: "10日後",
    14: "14日後",
    30: "30日後",
}

offset_labels = [
    offset_label_map.get(x, f"{x}日後")
    for x in available_offsets
]

label_to_offset = {
    offset_label_map.get(x, f"{x}日後"): x
    for x in available_offsets
}

st.sidebar.header("表示設定")

selected_label = st.sidebar.radio(
    "予測日",
    offset_labels,
    index=0,
)

selected_offset = label_to_offset[selected_label]

risk = risk_all[risk_all["offset_days"] == selected_offset].copy()

risk_mode = st.sidebar.radio(
    "表示するリスク",
    ["Core GAM risk", "Integrated risk", "MaxEnt suitability"],
    index=0,
)

show_context = st.sidebar.checkbox("環境proxyを表示", value=True)

has_kuroshio = "kuroshio_dist_km" in risk.columns and risk["kuroshio_dist_km"].notna().any()

target_date = str(risk["target_date"].iloc[0]) if "target_date" in risk.columns else "Unknown"
latest_jday = int(risk["Jday"].iloc[0]) if "Jday" in risk.columns else None

st.sidebar.markdown("---")
st.sidebar.write(f"Source: {source_file}")
st.sidebar.write(f"Target date: {target_date}")

if latest_jday is not None:
    st.sidebar.write(f"Jday: {latest_jday}")

if has_kuroshio:
    st.sidebar.success("Kuroshio context: active")
else:
    st.sidebar.info("Kuroshio context: not implemented")

if log is not None and len(log) > 0:
    st.sidebar.markdown("---")
    st.sidebar.write("Last update")
    st.sidebar.dataframe(log.tail(3), use_container_width=True)

if risk_mode == "Core GAM risk":
    value_col = "core_risk"
elif risk_mode == "Integrated risk":
    value_col = "integrated_risk"
else:
    value_col = "maxent_suitability" if "maxent_suitability" in risk.columns else "core_risk"

required_cols = ["NetID", "Latitude", "Longitude", value_col]
missing = [c for c in required_cols if c not in risk.columns]

if missing:
    st.error(f"risk csv に必要な列がありません: {missing}")
    st.stop()

risk[value_col] = pd.to_numeric(risk[value_col], errors="coerce")
risk = risk.dropna(subset=["Latitude", "Longitude", value_col])

top = risk.sort_values(value_col, ascending=False).iloc[0]

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Prediction", selected_label)
col2.metric("Top set net", top["net_name"])
col3.metric("Risk score", f"{top[value_col]:.3f}")

if "SST" in risk.columns:
    col4.metric("SST", f"{float(top['SST']):.1f} ℃")
else:
    col4.metric("SST", "NA")

if "depth_m" in risk.columns:
    col5.metric("Depth", f"{float(top['depth_m']):.0f} m")
else:
    col5.metric("Depth", "NA")

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

tooltip_html = "<b>定置網:</b> {net_name}<br>"
tooltip_html += "<b>NetID:</b> {NetID}<br>"
tooltip_html += "<b>予測日:</b> {target_date}<br>"
tooltip_html += f"<b>{value_col}:</b> " + "{" + value_col + "}<br>"

for c in [
    "core_risk",
    "maxent_suitability",
    "integrated_risk",
    "SST",
    "depth_m",
    "kuroshio_dist_km",
    "kuroshio_context",
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
    tooltip={
        "html": tooltip_html,
        "style": {"backgroundColor": "white", "color": "black"},
    },
)

st.pydeck_chart(deck, use_container_width=True)

st.markdown("### Risk ranking")

display_cols = [
    "net_name",
    "NetID",
    "target_date",
    "offset_days",
    "Jday",
    "SST",
    "SST_source",
    "depth_m",
    "core_risk",
    "core_risk_class",
    "maxent_suitability",
    "kuroshio_dist_km",
    "kuroshio_context",
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

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Upwelling proxy")

        if "upwelling_context" in risk.columns:
            up_cols = ["net_name", "NetID", "upwelling_context"]

            if "upwelling_proxy" in risk.columns:
                up_cols.insert(2, "upwelling_proxy")

            tmp = risk[up_cols].copy()
            tmp = tmp.sort_values("upwelling_context", ascending=False)

            for c in tmp.columns:
                if tmp[c].dtype.kind in "fc":
                    tmp[c] = tmp[c].round(3)

            st.dataframe(tmp, use_container_width=True, hide_index=True)
        else:
            st.info("upwelling_context がありません。")

    with c2:
        st.subheader("Kuroshio proxy")

        if has_kuroshio:
            tmp = risk[["net_name", "NetID", "kuroshio_dist_km", "kuroshio_context"]].copy()
            tmp = tmp.sort_values("kuroshio_context", ascending=False)

            for c in tmp.columns:
                if tmp[c].dtype.kind in "fc":
                    tmp[c] = tmp[c].round(3)

            st.dataframe(tmp, use_container_width=True, hide_index=True)
        else:
            st.info("黒潮距離データはまだ入っていません。")

st.markdown("---")
st.markdown("### Model note")

st.write(
    """
    Core risk は GAM による主リスクです。
    モデル式は `presence ~ s(Jday, bs="cc") + s(SST) + s(depth_m)` です。

    Integrated risk は実験的な総合スコアです。

    Integrated risk = 0.75 × Core GAM risk + 0.15 × Kuroshio context + 0.10 × Upwelling context

    現在の将来予測では、SSTは実測予報値ではなく、過去データから推定した同時期の季節値を使っています。
    そのため、3〜30日後の予測は「季節性にもとづく参考リスク」として解釈してください。

    Core GAM risk を主指標として扱い、Integrated risk は補助的に解釈してください。
    """
)
