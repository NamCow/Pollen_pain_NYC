"""
Pollen & Pain — NYC Asthma ED Surge Forecasting Dashboard

Streamlit application pulling all data from PostgreSQL (pollen schema).
Run: streamlit run src/dashboard/app.py
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
import json
import psycopg2
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Pollen & Pain — NYC Asthma Forecasting",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-warm: #FAF8F5;
    --bg-card: #FFFFFF;
    --text-primary: #1A1A1A;
    --text-secondary: #5A5A5A;
    --text-muted: #8A8A8A;
    --border: #E8E4DF;
    --accent-warm: #C4501A;
    --accent-gold: #B8860B;
    --risk-low: #2D7D4F;
    --risk-moderate: #D4A017;
    --risk-high: #D4652A;
    --risk-severe: #A8201A;
    --pollen-tree: #5B8C3E;
    --pollen-grass: #7CAE3A;
    --pollen-weed: #C4941A;
    --pollen-mold: #8B7355;
}

.stApp {
    background-color: var(--bg-warm) !important;
}

/* Override Streamlit defaults */
.stApp header { background-color: transparent !important; }

h1, h2, h3, h4 {
    font-family: 'Playfair Display', Georgia, 'Times New Roman', serif !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em !important;
}

h1 {
    font-size: 2.1rem !important;
    font-weight: 700 !important;
    line-height: 1.15 !important;
}

h2 {
    font-size: 1.35rem !important;
    font-weight: 600 !important;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin-top: 1.5rem !important;
}

h3 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
}

p, li, span, div, label, .stMarkdown {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #F5F2ED !important;
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] h1 {
    font-size: 1.4rem !important;
}

section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: var(--text-muted) !important;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px 18px;
}

div[data-testid="stMetric"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted) !important;
}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.6rem !important;
    color: var(--text-primary) !important;
}

/* Pollen strip */
.pollen-strip {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 24px;
    margin-bottom: 1.2rem;
}

.pollen-strip-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin-bottom: 8px;
}

.pollen-category {
    display: inline-block;
    margin-right: 28px;
}

.pollen-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--text-secondary);
}

.pollen-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 500;
}

/* Risk badges */
.risk-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 3px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.risk-low { background: #E8F5E9; color: #2D7D4F; }
.risk-moderate { background: #FFF8E1; color: #B8860B; }
.risk-high { background: #FBE9E7; color: #D4652A; }
.risk-severe { background: #FFEBEE; color: #A8201A; }

/* Detail panel */
.detail-panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px 24px;
    margin-top: 1rem;
}

.detail-panel h3 {
    font-family: 'Playfair Display', serif !important;
    margin-bottom: 4px;
}

.stat-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid #F0EDE8;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem;
}

.stat-label { color: var(--text-secondary); }
.stat-value { font-family: 'JetBrains Mono', monospace; font-weight: 500; }

/* Feature importance */
.feature-bar-container {
    margin: 6px 0;
}

.feature-name {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin-bottom: 2px;
}

.feature-bar-bg {
    background: #F0EDE8;
    border-radius: 2px;
    height: 6px;
    position: relative;
}

.feature-bar-fill {
    background: var(--accent-warm);
    border-radius: 2px;
    height: 6px;
    position: absolute;
    top: 0;
    left: 0;
}

/* Footer */
.footer {
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    color: var(--text-muted);
    line-height: 1.6;
}

/* Remove Streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Map container */
.map-container {
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)


# ── Database connection ──────────────────────────────────────────────────────

import os
from dotenv import load_dotenv

# Try to load .env from the parent directory
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

DB_PARAMS = dict(
    dbname=os.environ.get("DB_NAME", "postgres"),
    user=os.environ.get("DB_USER", "postgres"),
    password=os.environ.get("DB_PASSWORD", ""),
    host=os.environ.get("DB_HOST", "localhost"),
    port=int(os.environ.get("DB_PORT", 5432)),
    sslmode="require",
)
GEOJSON_PATH = Path("./data/Input/nta2020.geojson")


@st.cache_resource
def get_db_connection():
    return psycopg2.connect(**DB_PARAMS)


@st.cache_data(ttl=300)
def query_db(sql: str, params=None) -> pd.DataFrame:
    conn = psycopg2.connect(**DB_PARAMS)
    try:
        df = pd.read_sql_query(sql, conn, params=params)
        return df
    finally:
        conn.close()


@st.cache_data
def load_geojson():
    with open(GEOJSON_PATH) as f:
        return json.load(f)


@st.cache_data
def load_neighborhoods():
    return query_db("SELECT * FROM pollen.neighborhoods ORDER BY nta_code")


@st.cache_data
def load_modeling_table():
    return query_db("SELECT * FROM pollen.modeling_table ORDER BY nta_code, year_month")


@st.cache_data
def load_predictions():
    return query_db("SELECT * FROM pollen.model_predictions ORDER BY nta_code, year_month")


@st.cache_data
def load_feature_importance():
    return query_db("SELECT * FROM pollen.feature_importance ORDER BY importance DESC")


@st.cache_data
def load_complaints():
    return query_db("SELECT * FROM pollen.complaints_monthly ORDER BY borough, year_month")


@st.cache_data
def load_pollen_monthly():
    return query_db("SELECT * FROM pollen.pollen_monthly ORDER BY year_month")


# ── Helper functions ─────────────────────────────────────────────────────────

FEATURE_LABELS = {
    "chs_asthma_pct": "Neighborhood Asthma Prevalence",
    "pollen_28d_lag_avg": "Pollen Exposure (4-week lag)",
    "temp_min_mean": "Low Temperatures",
    "temp_max_mean": "High Temperatures",
    "total_dbh": "Tree Canopy Density",
    "pollen_14d_lag_avg": "Pollen Exposure (2-week lag)",
    "pollen_composite_avg": "Current Pollen Level",
    "pm25_mean": "Fine Particulate Matter (PM2.5)",
    "tree_avg": "Tree Pollen Count",
    "no2_mean": "Nitrogen Dioxide (NO₂)",
    "weed_avg": "Weed Pollen Count",
    "pct_good_health": "Tree Health Index",
    "grass_avg": "Grass Pollen Count",
    "mean_dbh": "Average Tree Size",
    "ozone_mean": "Ground-Level Ozone",
    "tree_max": "Peak Tree Pollen",
    "wind_max_mean": "Wind Speed",
    "tree_count": "Street Tree Count",
    "pollen_composite_max": "Peak Pollen Level",
    "precip_total": "Precipitation",
}

BOROUGH_MAP = {
    "Bronx": "BRONX",
    "Brooklyn": "BROOKLYN",
    "Manhattan": "MANHATTAN",
    "Queens": "QUEENS",
    "Staten Island": "STATEN ISLAND",
}


def classify_risk(ed_visits: float, low_thresh: float, high_thresh: float, severe_thresh: float) -> str:
    if ed_visits >= severe_thresh:
        return "Severe"
    elif ed_visits >= high_thresh:
        return "High"
    elif ed_visits >= low_thresh:
        return "Moderate"
    return "Low"


def risk_badge_html(risk_level: str) -> str:
    css_class = f"risk-{risk_level.lower()}"
    return f'<span class="risk-badge {css_class}">{risk_level}</span>'


def build_choropleth(geojson_data, month_data, selected_nta=None):
    m = folium.Map(
        location=[40.7128, -73.95],
        zoom_start=10.4,
        tiles="cartodbpositron",
        control_scale=False,
        zoom_control=False,
    )

    nta_values = dict(zip(month_data["nta_code"], month_data["ed_visits"]))
    nta_names = dict(zip(month_data["nta_code"], month_data["nta_name"]))
    nta_risks = dict(zip(month_data["nta_code"], month_data["risk_level"]))
    nta_boroughs = dict(zip(month_data["nta_code"], month_data["borough"]))
    valid_ntas = set(month_data["nta_code"])

    vmin = month_data["ed_visits"].quantile(0.05)
    vmax = month_data["ed_visits"].quantile(0.95)

    risk_colors = {"Low": "#2D7D4F", "Moderate": "#D4A017", "High": "#D4652A", "Severe": "#A8201A"}

    def style_function(feature):
        nta_code = feature["properties"]["NTA2020"]
        nta_type = feature["properties"]["NTAType"]

        if nta_type != "0" or nta_code not in valid_ntas:
            return {
                "fillColor": "#E8E4DF",
                "fillOpacity": 0.3,
                "color": "#D0CCC7",
                "weight": 0.5,
            }

        val = nta_values.get(nta_code, 0)
        risk = nta_risks.get(nta_code, "Low")
        color = risk_colors.get(risk, "#E8E4DF")

        norm = max(0, min(1, (val - vmin) / (vmax - vmin))) if vmax > vmin else 0.5
        opacity = 0.35 + 0.5 * norm

        is_selected = nta_code == selected_nta
        border_weight = 3 if is_selected else 0.8
        border_color = "#1A1A1A" if is_selected else "#FFFFFF"

        return {
            "fillColor": color,
            "fillOpacity": opacity,
            "color": border_color,
            "weight": border_weight,
        }

    def highlight_function(feature):
        return {
            "weight": 2.5,
            "color": "#1A1A1A",
            "fillOpacity": 0.85,
        }

    geojson_layer = folium.GeoJson(
        geojson_data,
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["NTA2020", "NTAName", "BoroName"],
            aliases=["Code:", "Neighborhood:", "Borough:"],
            style="font-family: 'DM Sans', sans-serif; font-size: 13px; padding: 8px 12px;",
        ),
    )

    for feature in geojson_data["features"]:
        nta_code = feature["properties"]["NTA2020"]
        if nta_code in valid_ntas:
            val = nta_values.get(nta_code, 0)
            risk = nta_risks.get(nta_code, "Low")
            name = nta_names.get(nta_code, "")
            feature["properties"]["ed_visits"] = f"{val:.1f}"
            feature["properties"]["risk"] = risk

    geojson_layer.add_to(m)
    return m


# ── Load all data ────────────────────────────────────────────────────────────

modeling = load_modeling_table()
predictions = load_predictions()
neighborhoods = load_neighborhoods()
feature_imp = load_feature_importance()
complaints = load_complaints()
pollen_monthly = load_pollen_monthly()
geojson_data = load_geojson()

available_months = sorted(modeling["year_month"].unique())

# Compute risk thresholds from overall distribution
ed_q25 = modeling["ed_visits"].quantile(0.25)
ed_q75 = modeling["ed_visits"].quantile(0.75)
ed_q90 = modeling["ed_visits"].quantile(0.90)

modeling["risk_level"] = modeling["ed_visits"].apply(
    lambda x: classify_risk(x, ed_q25, ed_q75, ed_q90)
)


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <span style="font-family: 'Playfair Display', serif; font-size: 1.5rem; font-weight: 700; color: #1A1A1A; line-height: 1.2;">
            Pollen &<br>Pain
        </span>
        <div style="font-family: 'DM Sans', sans-serif; font-size: 0.78rem; color: #8A8A8A; margin-top: 4px; letter-spacing: 0.03em;">
            NYC Asthma ED Forecasting
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    selected_month = st.selectbox(
        "Month",
        options=available_months,
        index=len(available_months) - 1,
        format_func=lambda x: pd.Timestamp(x + "-01").strftime("%B %Y"),
    )

    borough_options = ["All Boroughs"] + sorted(modeling["borough"].unique())
    selected_borough = st.selectbox("Borough", options=borough_options)

    risk_options = ["All Levels", "Low", "Moderate", "High", "Severe"]
    selected_risk = st.selectbox("Risk Level", options=risk_options)

    st.markdown("---")

    st.markdown("""
    <div style="font-family: 'DM Sans', sans-serif; font-size: 0.78rem; color: #8A8A8A; line-height: 1.6;">
        <strong style="color: #5A5A5A;">About</strong><br>
        Predicts monthly asthma emergency department visit counts at the
        neighborhood level across all five NYC boroughs.<br><br>
        <strong style="color: #5A5A5A;">Model</strong><br>
        XGBoost regression with walk-forward time-series validation.
        Trained on pollen, weather, air quality, tree canopy,
        and community health data.<br><br>
        <strong style="color: #5A5A5A;">Team</strong><br>
        Saketh Boddu · Andre Nguyen · Nam Lai
    </div>
    """, unsafe_allow_html=True)


# ── Filter data for selected month ──────────────────────────────────────────

month_data = modeling[modeling["year_month"] == selected_month].copy()
month_data["risk_level"] = month_data["ed_visits"].apply(
    lambda x: classify_risk(x, ed_q25, ed_q75, ed_q90)
)

if selected_borough != "All Boroughs":
    month_data = month_data[month_data["borough"] == selected_borough]

if selected_risk != "All Levels":
    month_data = month_data[month_data["risk_level"] == selected_risk]


# ── Pollen strip ─────────────────────────────────────────────────────────────

pollen_row = pollen_monthly[pollen_monthly["year_month"] == selected_month]

if not pollen_row.empty:
    pr = pollen_row.iloc[0]
    pollen_html = f"""
    <div class="pollen-strip">
        <div class="pollen-strip-title">Pollen Index · {pd.Timestamp(selected_month + "-01").strftime("%B %Y")} · City-Wide</div>
        <div>
            <span class="pollen-category">
                <span class="pollen-label" style="color: var(--pollen-tree);">● Tree</span>
                <span class="pollen-value" style="color: var(--pollen-tree);"> {pr['tree_avg']:.0f}</span>
            </span>
            <span class="pollen-category">
                <span class="pollen-label" style="color: var(--pollen-weed);">● Weed</span>
                <span class="pollen-value" style="color: var(--pollen-weed);"> {pr['weed_avg']:.1f}</span>
            </span>
            <span class="pollen-category">
                <span class="pollen-label" style="color: var(--pollen-grass);">● Grass</span>
                <span class="pollen-value" style="color: var(--pollen-grass);"> {pr['grass_avg']:.1f}</span>
            </span>
            <span class="pollen-category">
                <span class="pollen-label" style="color: var(--pollen-mold);">● Mold</span>
                <span class="pollen-value" style="color: var(--pollen-mold);"> {pr['mold_avg']:.1f}</span>
            </span>
            <span class="pollen-category" style="margin-left: 20px; padding-left: 20px; border-left: 1px solid var(--border);">
                <span class="pollen-label">Composite</span>
                <span class="pollen-value" style="color: var(--accent-warm);"> {pr['pollen_composite_avg']:.0f}</span>
                <span class="pollen-label" style="font-size: 0.72rem;"> avg</span>
                <span class="pollen-value" style="color: var(--accent-warm);"> {pr['pollen_composite_max']:.0f}</span>
                <span class="pollen-label" style="font-size: 0.72rem;"> peak</span>
            </span>
        </div>
    </div>
    """
    st.markdown(pollen_html, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="pollen-strip">
        <div class="pollen-strip-title">Pollen Index · Off-Season</div>
        <div style="font-family: 'DM Sans', sans-serif; font-size: 0.88rem; color: var(--text-muted);">
            No pollen data for this month — outside monitoring season (March–October)
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Summary metrics ──────────────────────────────────────────────────────────

display_month = pd.Timestamp(selected_month + "-01").strftime("%B %Y")
st.markdown(f"## {display_month}")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Neighborhoods", f"{len(month_data)}")
with col2:
    st.metric("Avg ED Visits", f"{month_data['ed_visits'].mean():.1f}")
with col3:
    severe_count = len(month_data[month_data["risk_level"] == "Severe"])
    st.metric("Severe Risk", f"{severe_count}")
with col4:
    high_count = len(month_data[month_data["risk_level"] == "High"])
    st.metric("High Risk", f"{high_count}")
with col5:
    total_ed = month_data["ed_visits"].sum()
    st.metric("Total ED Visits", f"{total_ed:,.0f}")


# ── Main layout: Map + Detail ───────────────────────────────────────────────

# Initialize session state for selected NTA
if "selected_nta" not in st.session_state:
    st.session_state.selected_nta = None

map_col, detail_col = st.columns([3, 2])

with map_col:
    st.markdown("### Neighborhood Risk Map")

    choropleth = build_choropleth(geojson_data, month_data, st.session_state.selected_nta)
    map_output = st_folium(choropleth, width=None, height=520, returned_objects=["last_active_drawing", "last_object_clicked_tooltip"])

    if map_output and map_output.get("last_object_clicked_tooltip"):
        tooltip_text = str(map_output["last_object_clicked_tooltip"])
        for _, row in month_data.iterrows():
            if row["nta_code"] in tooltip_text or row["nta_name"] in tooltip_text:
                st.session_state.selected_nta = row["nta_code"]
                break

    # Legend
    st.markdown("""
    <div style="display: flex; gap: 18px; margin-top: 8px; font-family: 'DM Sans', sans-serif; font-size: 0.78rem;">
        <span><span style="display:inline-block; width:12px; height:12px; background:#2D7D4F; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>Low</span>
        <span><span style="display:inline-block; width:12px; height:12px; background:#D4A017; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>Moderate</span>
        <span><span style="display:inline-block; width:12px; height:12px; background:#D4652A; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>High</span>
        <span><span style="display:inline-block; width:12px; height:12px; background:#A8201A; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>Severe</span>
        <span><span style="display:inline-block; width:12px; height:12px; background:#E8E4DF; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>Non-residential / Excluded</span>
    </div>
    """, unsafe_allow_html=True)


with detail_col:
    if st.session_state.selected_nta:
        nta_code = st.session_state.selected_nta
        nta_row = month_data[month_data["nta_code"] == nta_code]

        if not nta_row.empty:
            nta = nta_row.iloc[0]
            risk_html = risk_badge_html(nta["risk_level"])

            st.markdown(f"""
            <div class="detail-panel">
                <h3>{nta['nta_name']}</h3>
                <div style="font-family: 'DM Sans', sans-serif; font-size: 0.82rem; color: var(--text-muted); margin-bottom: 12px;">
                    {nta['borough']} · {nta_code} {risk_html}
                </div>
                <div class="stat-row">
                    <span class="stat-label">Est. ED Visits</span>
                    <span class="stat-value">{nta['ed_visits']:.1f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Asthma Prevalence</span>
                    <span class="stat-value">{nta['chs_asthma_pct']:.1f}%</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Street Trees</span>
                    <span class="stat-value">{int(nta['tree_count']):,}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Tree Health (% Good)</span>
                    <span class="stat-value">{nta['pct_good_health']*100:.0f}%</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Avg Temperature</span>
                    <span class="stat-value">{nta['temp_max_mean']:.0f}°C / {nta['temp_min_mean']:.0f}°C</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">PM2.5</span>
                    <span class="stat-value">{nta['pm25_mean']:.1f} µg/m³</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 12-month history chart
            st.markdown("#### ED Visit History")
            nta_history = modeling[modeling["nta_code"] == nta_code].sort_values("year_month").tail(12)
            nta_preds = predictions[predictions["nta_code"] == nta_code].sort_values("year_month")

            if not nta_history.empty:
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=nta_history["year_month"],
                    y=nta_history["ed_visits"],
                    mode="lines+markers",
                    name="Actual",
                    line=dict(color="#1A1A1A", width=2),
                    marker=dict(size=5, color="#1A1A1A"),
                ))

                if not nta_preds.empty:
                    latest_fold = nta_preds["fold"].max()
                    fold_preds = nta_preds[nta_preds["fold"] == latest_fold].tail(12)
                    fig.add_trace(go.Scatter(
                        x=fold_preds["year_month"],
                        y=fold_preds["pred"],
                        mode="lines+markers",
                        name="Predicted",
                        line=dict(color="#C4501A", width=2, dash="dot"),
                        marker=dict(size=5, color="#C4501A"),
                    ))

                fig.update_layout(
                    height=240,
                    margin=dict(l=0, r=0, t=10, b=30),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Sans", size=12, color="#5A5A5A"),
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02,
                        xanchor="left", x=0, font=dict(size=11)
                    ),
                    xaxis=dict(
                        showgrid=False, tickfont=dict(size=10),
                        tickangle=-45,
                    ),
                    yaxis=dict(
                        showgrid=True, gridcolor="#F0EDE8",
                        title=None, tickfont=dict(size=10),
                    ),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # 311 complaints for this borough
            boro_311 = BOROUGH_MAP.get(nta["borough"], "")
            if boro_311:
                boro_complaints = complaints[complaints["borough"] == boro_311].sort_values("year_month").tail(12)
                if not boro_complaints.empty:
                    st.markdown("#### Air Quality Complaints (311)")
                    fig311 = go.Figure()
                    fig311.add_trace(go.Bar(
                        x=boro_complaints["year_month"],
                        y=boro_complaints["complaint_count"],
                        marker_color="#8B7355",
                        opacity=0.7,
                    ))
                    fig311.update_layout(
                        height=180,
                        margin=dict(l=0, r=0, t=10, b=30),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="DM Sans", size=11, color="#5A5A5A"),
                        showlegend=False,
                        xaxis=dict(showgrid=False, tickfont=dict(size=10), tickangle=-45),
                        yaxis=dict(showgrid=True, gridcolor="#F0EDE8", title=None, tickfont=dict(size=10)),
                    )
                    st.plotly_chart(fig311, use_container_width=True, config={"displayModeBar": False})

        else:
            st.info("Selected neighborhood not visible with current filters.")
    else:
        st.markdown("""
        <div class="detail-panel" style="text-align: center; padding: 60px 24px;">
            <div style="font-family: 'Playfair Display', serif; font-size: 1.1rem; color: var(--text-secondary); margin-bottom: 8px;">
                Select a Neighborhood
            </div>
            <div style="font-family: 'DM Sans', sans-serif; font-size: 0.85rem; color: var(--text-muted);">
                Click on a colored area on the map to view<br>detailed statistics and trend history.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Feature Importance ───────────────────────────────────────────────────────

st.markdown("## What's Driving Predictions")
st.markdown("""
<div style="font-family: 'DM Sans', sans-serif; font-size: 0.88rem; color: var(--text-secondary); margin-bottom: 1rem; max-width: 640px;">
    The model weighs multiple environmental and community health signals to estimate ED visit risk.
    Bars show each factor's relative contribution to predictions across all neighborhoods.
</div>
""", unsafe_allow_html=True)

fi_col1, fi_col2 = st.columns(2)

top_features = feature_imp.head(10)
max_imp = top_features["importance"].max()

left_features = top_features.head(5)
right_features = top_features.tail(5)

for col, features in [(fi_col1, left_features), (fi_col2, right_features)]:
    with col:
        for _, row in features.iterrows():
            label = FEATURE_LABELS.get(row["feature"], row["feature"])
            pct = row["importance"] / max_imp * 100
            imp_pct = row["importance"] * 100

            st.markdown(f"""
            <div class="feature-bar-container">
                <div class="feature-name">{label} <span style="font-family: 'JetBrains Mono'; font-size: 0.72rem; color: var(--text-muted);">{imp_pct:.1f}%</span></div>
                <div class="feature-bar-bg">
                    <div class="feature-bar-fill" style="width: {pct}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ── Borough comparison ───────────────────────────────────────────────────────

st.markdown("## Borough Overview")

borough_summary = (
    month_data.groupby("borough")
    .agg(
        avg_ed=("ed_visits", "mean"),
        total_ed=("ed_visits", "sum"),
        nta_count=("nta_code", "nunique"),
        severe=("risk_level", lambda x: (x == "Severe").sum()),
        high=("risk_level", lambda x: (x == "High").sum()),
    )
    .reset_index()
    .sort_values("avg_ed", ascending=False)
)

fig_boro = go.Figure()
colors = ["#C4501A", "#D4A017", "#2D7D4F", "#5A7D9A", "#8B7355"]

for i, (_, row) in enumerate(borough_summary.iterrows()):
    fig_boro.add_trace(go.Bar(
        x=[row["borough"]],
        y=[row["avg_ed"]],
        name=row["borough"],
        marker_color=colors[i % len(colors)],
        text=f"{row['avg_ed']:.1f}",
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=12),
        showlegend=False,
    ))

fig_boro.update_layout(
    height=300,
    margin=dict(l=0, r=0, t=20, b=40),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", size=12, color="#5A5A5A"),
    xaxis=dict(showgrid=False, tickfont=dict(size=12, family="DM Sans")),
    yaxis=dict(showgrid=True, gridcolor="#F0EDE8", title="Avg. ED Visits per Neighborhood", title_font=dict(size=11)),
    bargap=0.4,
)

st.plotly_chart(fig_boro, use_container_width=True, config={"displayModeBar": False})


# ── Top risk neighborhoods table ─────────────────────────────────────────────

st.markdown("## Highest-Risk Neighborhoods")

top_risk = (
    month_data
    .nlargest(15, "ed_visits")
    [["nta_name", "borough", "ed_visits", "risk_level", "chs_asthma_pct", "tree_count"]]
    .copy()
)
top_risk.columns = ["Neighborhood", "Borough", "Est. ED Visits", "Risk", "Asthma Prev. (%)", "Street Trees"]
top_risk["Est. ED Visits"] = top_risk["Est. ED Visits"].round(1)
top_risk["Asthma Prev. (%)"] = top_risk["Asthma Prev. (%)"].round(1)
top_risk = top_risk.reset_index(drop=True)
top_risk.index = top_risk.index + 1

st.dataframe(
    top_risk,
    use_container_width=True,
    height=400,
)


# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="footer">
    <strong>Pollen & Pain</strong> — Forecasting Neighborhood-Level Asthma Emergency Department Surges in New York City<br>
    Data sources: NYC DOHMH Syndromic Surveillance, AAAAI National Allergy Bureau, Open-Meteo, EPA AQS, NYC Street Tree Census, NYC 311<br>
    Model: XGBoost regression with walk-forward time-series cross-validation · 197 NTAs · March 2022–present<br>
    Built by Saketh Boddu, Andre Nguyen, and Nam Lai
</div>
""", unsafe_allow_html=True)
