
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
from urllib.parse import urlencode


APP_ROOT = Path(__file__).resolve().parent

FORECAST_PATH = APP_ROOT / "outputs" / "forecast_risk.csv"
RISK_PATH = APP_ROOT / "outputs" / "latest_risk.csv"
LOG_PATH = APP_ROOT / "outputs" / "update_log.csv"

HIST_DAILY_PATH = APP_ROOT / "outputs" / "historical_risk_daily.csv"
HIST_MONTHLY_PATH = APP_ROOT / "outputs" / "historical_risk_monthly.csv"
HIST_YEARLY_PATH = APP_ROOT / "outputs" / "historical_risk_yearly.csv"

# Fixed map extent covering Kochi and the adjacent Pacific slope.  The risk
# points retain their J-EGG500 depth values; this image is only a regional
# bathymetric backdrop to make the seafloor gradient visible.
BATHYMETRY_BOUNDS = [131.5, 31.5, 135.5, 34.5]  # west, south, east, north
GEBCO_WMS_URL = "https://wms.gebco.net/mapserv"
GEBCO_ATTRIBUTION = (
    "Imagery reproduced from the GEBCO_2026 Grid, "
    "GEBCO Bathymetric Compilation Group (2026)."
)

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
    for c in ["relative_risk_percentile", "core_percentile"]:
        if c in df.columns:
            return c
    return None


def get_class_col(df):
    for c in ["relative_risk_class", "core_risk_class"]:
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
            st.metric("平均相対順位", f"{pd.to_numeric(df[risk_col], errors='coerce').mean():.1%}")
        else:
            st.metric("平均相対順位", "NA")

    with c3:
        if risk_col is not None:
            st.metric("最大相対順位", f"{pd.to_numeric(df[risk_col], errors='coerce').max():.1%}")
        else:
            st.metric("最大相対順位", "NA")

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

    risk_layer = pdk.Layer(
        "ScatterplotLayer",
        data=d,
        get_position="[Longitude, Latitude]",
        get_fill_color="[r, g, b, a]",
        get_line_color=[0, 0, 0, 160],
        get_radius="radius",
        pickable=True,
        auto_highlight=True,
    )

    show_bathymetry = st.toggle(
        "海底水深の勾配を表示",
        value=True,
        key=f"bathymetry_{title}",
        help="背景はGEBCO 2026、各定置網の解析用水深はJ-EGG500です。",
    )

    layers = []
    if show_bathymetry:
        wms_params = {
            "service": "WMS",
            "version": "1.1.1",
            "request": "GetMap",
            "layers": "GEBCO_LATEST",
            "styles": "default",
            "format": "image/png",
            "transparent": "true",
            "srs": "EPSG:4326",
            "bbox": ",".join(map(str, BATHYMETRY_BOUNDS)),
            "width": 1400,
            "height": 1050,
        }
        bathymetry_url = f"{GEBCO_WMS_URL}?{urlencode(wms_params)}"
        layers.append(
            pdk.Layer(
                "BitmapLayer",
                data=None,
                image=bathymetry_url,
                bounds=BATHYMETRY_BOUNDS,
                opacity=0.72,
            )
        )

    # Always draw risk points last so they remain legible above bathymetry.
    layers.append(risk_layer)

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
        Relative percentile: {_risk_value}<br/>
        Class: {_risk_class}<br/>
        SST: {SST}<br/>
        SST anomaly: {SST_anomaly} °C<br/>
        J-EGG500 depth: {depth_m} m
        """,
        "style": {
            "backgroundColor": "white",
            "color": "black"
        }
    }

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    )

    st.subheader(title)
    st.pydeck_chart(deck, use_container_width=True)
    if show_bathymetry:
        st.caption(
            "海底地形背景：GEBCO 2026（青が濃いほど深い）。"
            "地点を選択すると解析に用いたJ-EGG500水深を確認できます。 "
            + GEBCO_ATTRIBUTION
        )


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
        "SST_climatology",
        "SST_anomaly",
        "depth_m",
        "GEBCO_old_m",
        "depth_JEGG500_IDW_m",
        "JEGG_quality",
        "CHL_monthly",
        "CHL_log10",
        "core_risk",
        "core_percentile",
        "core_risk_class",
        "relative_risk_percentile",
        "relative_risk_class",
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


def uses_current_model(df):
    if df.empty or "model_main" not in df.columns:
        return False
    labels = df["model_main"].dropna().astype(str)
    return (labels.str.contains("SST anomaly", case=False)).all() if len(labels) else False


def show_current_forecast_tab(forecast, latest):
    st.header("現在・予報リスク")
    st.info("表示値は固定した学習データ内での相対順位です。出現確率そのものではありません。")

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
    if not uses_current_model(hist_daily):
        st.warning("履歴データは旧モデル版のため非表示です。GitHub Actions の次回実行後に更新されます。")
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

    st.caption("OISST偏差、周期Jday、J-EGG500水深による日別の相対リスク順位。")
    show_metric_cards(show)
    show_risk_map(show, f"Historical daily risk: {selected_date}")
    show_table(show)


def show_historical_monthly_tab(hist_monthly):
    st.header("過去予測：月別")

    if hist_monthly.empty:
        st.warning("historical_risk_monthly.csv が見つかりません。")
        return
    if not uses_current_model(hist_monthly):
        st.warning("履歴データは旧モデル版のため非表示です。GitHub Actions の次回実行後に更新されます。")
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

    st.caption("日別の相対リスク順位を月単位で平均した参考値です。")
    show_metric_cards(show)
    show_risk_map(show, f"Historical monthly risk: {selected_year}-{selected_month:02d}")
    show_table(show)


def show_historical_yearly_tab(hist_yearly):
    st.header("過去予測：年別")

    if hist_yearly.empty:
        st.warning("historical_risk_yearly.csv が見つかりません。")
        return
    if not uses_current_model(hist_yearly):
        st.warning("履歴データは旧モデル版のため非表示です。GitHub Actions の次回実行後に更新されます。")
        return

    df = hist_yearly.copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    years = sorted(df["Year"].dropna().astype(int).unique())

    selected_year = st.selectbox("年を選択", years, index=len(years) - 1, key="hist_year_year_select")

    show = df[df["Year"] == selected_year].copy()

    st.caption("日別の相対リスク順位を年単位で平均した参考値です。")
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
st.caption("高知県沿岸の定置網におけるジンベエザメ出現リスクモニター v0.6")

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
        - 主モデル：**周期Jday + SST偏差 + J-EGG500 IDW水深**（R `mgcv`, REML, `select=TRUE`）
        - 表示値：固定学習データに対する**相対順位（percentile）**。絶対的な出現確率ではありません。
        - 検証結果：見かけのAUC 0.904、LONO AUC 0.858 ± 0.075、AIC 1349.0、逸脱度説明率34.7%。
        - 生SSTはJdayとのconcurvityが高かったため主モデルから除外し、SST偏差へ変更しました。
        - 生SSTのみの感度解析では最適域は約23.2℃でした。
        - 月別・年別値は日別相対順位の平均で、参考的な集約値です。
        """
    )
    show_update_log()
