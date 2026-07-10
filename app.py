
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk


APP_ROOT = Path(__file__).resolve().parent

FORECAST_PATH = APP_ROOT / "outputs" / "forecast_risk.csv"
RISK_PATH = APP_ROOT / "outputs" / "latest_risk.csv"
LOG_PATH = APP_ROOT / "outputs" / "update_log.csv"

HIST_DAILY_PATH = APP_ROOT / "outputs" / "historical_risk_daily.csv"
HIST_MONTHLY_PATH = APP_ROOT / "outputs" / "historical_risk_monthly.csv"
HIST_YEARLY_PATH = APP_ROOT / "outputs" / "historical_risk_yearly.csv"

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


@st.cache_data
def read_csv_if_exists(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_all_data():
    forecast = read_csv_if_exists(FORECAST_PATH)
    latest = read_csv_if_exists(RISK_PATH)
    hist_daily = read_csv_if_exists(HIST_DAILY_PATH)
    hist_monthly = read_csv_if_exists(HIST_MONTHLY_PATH)
    hist_yearly = read_csv_if_exists(HIST_YEARLY_PATH)
    return forecast, latest, hist_daily, hist_monthly, hist_yearly


def risk_class_to_color(cls):
    cls = str(cls)

    if cls == "Very high":
        return [215, 25, 28, 210]
    if cls == "High":
        return [253, 174, 97, 210]
    if cls == "Moderate":
        return [255, 255, 191, 210]
    if cls == "Low":
        return [145, 191, 219, 210]

    return [180, 180, 180, 180]


def get_risk_col(df):
    for c in ["integrated_risk", "core_risk", "maxent_suitability"]:
        if c in df.columns:
            return c
    return None


def get_class_col(df):
    for c in ["integrated_risk_class", "core_risk_class"]:
        if c in df.columns:
            return c
    return None


def prepare_map_df(df):
    d = df.copy()

    if "Latitude" not in d.columns or "Longitude" not in d.columns:
        return pd.DataFrame()

    d["Latitude"] = pd.to_numeric(d["Latitude"], errors="coerce")
    d["Longitude"] = pd.to_numeric(d["Longitude"], errors="coerce")

    d = d.dropna(subset=["Latitude", "Longitude"]).copy()

    risk_col = get_risk_col(d)
    class_col = get_class_col(d)

    if risk_col is None:
        d["_risk_value"] = np.nan
    else:
        d["_risk_value"] = pd.to_numeric(d[risk_col], errors="coerce")

    if class_col is None:
        d["_risk_class"] = "Unknown"
    else:
        d["_risk_class"] = d[class_col].fillna("Unknown").astype(str)

    color_values = d["_risk_class"].apply(risk_class_to_color)
    color_df = pd.DataFrame(color_values.tolist(), columns=["r", "g", "b", "a"], index=d.index)

    d = pd.concat([d, color_df], axis=1)

    risk_for_radius = d["_risk_value"].fillna(0).clip(0, 1)
    d["radius"] = 3500 + risk_for_radius * 8500

    if "net_label" not in d.columns:
        if "NetID" in d.columns:
            d["net_label"] = "NetID " + d["NetID"].astype(str)
        else:
            d["net_label"] = "Unknown net"

    if "date" not in d.columns:
        if "target_date" in d.columns:
            d["date"] = d["target_date"]
        else:
            d["date"] = ""

    return d


def show_metric_cards(df):
    risk_col = get_risk_col(df)
    class_col = get_class_col(df)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("表示地点数", f"{len(df):,}")

    with c2:
        if risk_col is not None:
            st.metric("平均リスク", f"{pd.to_numeric(df[risk_col], errors='coerce').mean():.3f}")
        else:
            st.metric("平均リスク", "NA")

    with c3:
        if risk_col is not None:
            st.metric("最大リスク", f"{pd.to_numeric(df[risk_col], errors='coerce').max():.3f}")
        else:
            st.metric("最大リスク", "NA")

    with c4:
        if class_col is not None:
            vc = df[class_col].value_counts()
            top_class = vc.index[0] if len(vc) > 0 else "NA"
            st.metric("最多リスク区分", str(top_class))
        else:
            st.metric("最多リスク区分", "NA")


def show_risk_map(df, title="Risk map"):
    d = prepare_map_df(df)

    if d.empty:
        st.warning("地図表示に必要な Latitude / Longitude がありません。")
        return

    center_lat = float(d["Latitude"].mean())
    center_lon = float(d["Longitude"].mean())

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=d,
        get_position="[Longitude, Latitude]",
        get_fill_color="[r, g, b, a]",
        get_line_color=[0, 0, 0, 160],
        get_radius="radius",
        pickable=True,
        auto_highlight=True,
    )

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=7.3,
        pitch=0,
    )

    tooltip = {
        "html": """
        <b>{net_label}</b><br/>
        Date: {date}<br/>
        Risk: {_risk_value}<br/>
        Class: {_risk_class}<br/>
        SST: {SST}<br/>
        Depth: {depth_m} m<br/>
        CHL: {CHL_monthly} mg m⁻³
        """,
        "style": {
            "backgroundColor": "white",
            "color": "black"
        }
    }

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    )

    st.subheader(title)
    st.pydeck_chart(deck, use_container_width=True)


def show_table(df):
    cols = [
        "date",
        "target_date",
        "period_label",
        "forecast_label",
        "NetID",
        "net_name",
        "net_label",
        "Latitude",
        "Longitude",
        "Jday",
        "SST",
        "depth_m",
        "CHL_monthly",
        "CHL_log10",
        "core_risk",
        "core_percentile",
        "core_risk_class",
        "maxent_suitability",
        "integrated_risk",
        "integrated_percentile_today",
        "integrated_risk_class",
        "model_main",
        "note",
    ]

    use_cols = [c for c in cols if c in df.columns]

    risk_col = get_risk_col(df)

    if risk_col is not None:
        show_df = df.sort_values(risk_col, ascending=False)
    else:
        show_df = df.copy()

    st.dataframe(show_df[use_cols], use_container_width=True, hide_index=True)


def parse_date_col(df, col="date"):
    d = df.copy()

    if col not in d.columns:
        if "target_date" in d.columns:
            col = "target_date"
        elif "Date" in d.columns:
            col = "Date"
        else:
            return d, None

    d["_date"] = pd.to_datetime(d[col], errors="coerce").dt.date

    return d, "_date"


def show_current_forecast_tab(forecast, latest):
    st.header("現在・予報リスク")

    if not forecast.empty:
        df, date_col = parse_date_col(forecast, "target_date")

        st.caption("forecast_risk.csv を表示しています。")

        if date_col is not None:
            dates = sorted(df[date_col].dropna().unique())

            selected_date = st.selectbox(
                "表示日を選択",
                dates,
                index=0,
                key="forecast_date_select"
            )

            show = df[df[date_col] == selected_date].copy()
        else:
            show = df.copy()

        if "forecast_label" in show.columns:
            labels = show["forecast_label"].dropna().unique().tolist()
            st.caption(" / ".join(map(str, labels)))

        show_metric_cards(show)
        show_risk_map(show, "Current / forecast whale shark risk")
        show_table(show)

    elif not latest.empty:
        st.caption("latest_risk.csv を表示しています。")
        show = latest.copy()
        show_metric_cards(show)
        show_risk_map(show, "Latest whale shark risk")
        show_table(show)

    else:
        st.warning("forecast_risk.csv または latest_risk.csv が見つかりません。")


def show_historical_daily_tab(hist_daily):
    st.header("過去予測：日別")

    if hist_daily.empty:
        st.warning("historical_risk_daily.csv が見つかりません。")
        return

    df, date_col = parse_date_col(hist_daily, "date")

    dates = sorted(df[date_col].dropna().unique())

    default_index = len(dates) - 1

    selected_date = st.date_input(
        "日付を選択",
        value=dates[default_index],
        min_value=dates[0],
        max_value=dates[-1],
        key="hist_daily_date_input"
    )

    show = df[df[date_col] == selected_date].copy()

    st.caption("Historical daily risk based on OISST, Jday, and depth.")
    show_metric_cards(show)
    show_risk_map(show, f"Historical daily risk: {selected_date}")
    show_table(show)


def show_historical_monthly_tab(hist_monthly):
    st.header("過去予測：月別")

    if hist_monthly.empty:
        st.warning("historical_risk_monthly.csv が見つかりません。")
        return

    df = hist_monthly.copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce").astype("Int64")

    years = sorted(df["Year"].dropna().astype(int).unique())

    c1, c2 = st.columns(2)

    with c1:
        selected_year = st.selectbox("年を選択", years, index=len(years) - 1, key="hist_month_year_select")

    months = sorted(df[df["Year"] == selected_year]["Month"].dropna().astype(int).unique())

    with c2:
        selected_month = st.selectbox("月を選択", months, index=0, key="hist_month_month_select")

    show = df[
        (df["Year"] == selected_year) &
        (df["Month"] == selected_month)
    ].copy()

    st.caption("Monthly historical risk is the mean of daily historical risks.")
    show_metric_cards(show)
    show_risk_map(show, f"Historical monthly risk: {selected_year}-{selected_month:02d}")
    show_table(show)


def show_historical_yearly_tab(hist_yearly):
    st.header("過去予測：年別")

    if hist_yearly.empty:
        st.warning("historical_risk_yearly.csv が見つかりません。")
        return

    df = hist_yearly.copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    years = sorted(df["Year"].dropna().astype(int).unique())

    selected_year = st.selectbox("年を選択", years, index=len(years) - 1, key="hist_year_year_select")

    show = df[df["Year"] == selected_year].copy()

    st.caption("Yearly historical risk is the mean of daily historical risks.")
    show_metric_cards(show)
    show_risk_map(show, f"Historical yearly risk: {selected_year}")
    show_table(show)


def show_update_log():
    with st.expander("Update log / data status"):
        if LOG_PATH.exists():
            log = pd.read_csv(LOG_PATH)
            st.dataframe(log, use_container_width=True, hide_index=True)
        else:
            st.caption("update_log.csv が見つかりません。")

        st.write("Files")
        files = {
            "forecast_risk.csv": FORECAST_PATH.exists(),
            "latest_risk.csv": RISK_PATH.exists(),
            "historical_risk_daily.csv": HIST_DAILY_PATH.exists(),
            "historical_risk_monthly.csv": HIST_MONTHLY_PATH.exists(),
            "historical_risk_yearly.csv": HIST_YEARLY_PATH.exists(),
        }
        st.json(files)


check_password()

st.title("Kochi Whale Shark Risk Monitor")
st.caption("高知県沿岸の定置網におけるジンベエザメ出現リスクモニター v0.4")

forecast, latest, hist_daily, hist_monthly, hist_yearly = load_all_data()

tab_now, tab_hist_day, tab_hist_month, tab_hist_year, tab_about = st.tabs([
    "現在・予報",
    "過去 日別",
    "過去 月別",
    "過去 年別",
    "説明・ログ",
])

with tab_now:
    show_current_forecast_tab(forecast, latest)

with tab_hist_day:
    show_historical_daily_tab(hist_daily)

with tab_hist_month:
    show_historical_monthly_tab(hist_monthly)

with tab_hist_year:
    show_historical_yearly_tab(hist_yearly)

with tab_about:
    st.header("Model notes")
    st.markdown(
        """
        - Main model: **GAM: Jday + SST + depth**
        - Historical risk: based on historical **NOAA OISST**, Julian day, and depth.
        - Historical daily risk is calculated for fixed set-net locations.
        - Monthly and yearly historical risks are averages of daily historical risks.
        - Kuroshio and upwelling contexts are not included in the historical mode at this stage.
        """
    )
    show_update_log()
