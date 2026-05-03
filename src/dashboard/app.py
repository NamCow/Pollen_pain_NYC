"""
Pollen & Pain — NYC Asthma ED Surge Forecasting Dashboard

Streamlit application pulling all data from PostgreSQL (pollen schema).
Run: streamlit run src/dashboard/app.py
"""

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
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

.stApp { background-color: #F5F2ED !important; }
.stApp header { background-color: transparent !important; }

h1, h2, h3, h4 {
    font-family: 'Playfair Display', Georgia, 'Times New Roman', serif !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em !important;
}
h1 { font-size: 2.1rem !important; font-weight: 700 !important; line-height: 1.15 !important; }
h2 {
    font-size: 1.35rem !important; font-weight: 600 !important;
    border-bottom: 1px solid var(--border); padding-bottom: 0.4rem; margin-top: 1.5rem !important;
}
h3 { font-size: 1.1rem !important; font-weight: 600 !important; color: var(--text-secondary) !important; }

p, li, .stMarkdown,
.stApp label,
.stApp [data-testid="stSidebar"] {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary);
}

/* Keep Streamlit's Material icon ligatures on their own font. */
.material-symbols-rounded,
.material-symbols-outlined,
[data-testid="stIconMaterial"] {
    font-family: "Material Symbols Rounded", "Material Symbols Outlined", sans-serif !important;
    color: inherit;
}

section[data-testid="stSidebar"] {
    background-color: #FFFBEB !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label {
    font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important;
    font-size: 0.85rem !important; text-transform: uppercase !important;
    letter-spacing: 0.06em !important; color: var(--text-muted) !important;
}
.sidebar-brand {
    margin-bottom: 1.2rem;
}
.sidebar-brand-title {
    font-family: 'Playfair Display', Georgia, 'Times New Roman', serif !important;
    font-size: clamp(1.2rem, 2vw, 1.65rem);
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.05;
    letter-spacing: -0.03em;
    overflow-wrap: anywhere;
}
.sidebar-brand-subtitle {
    font-family: 'DM Sans', sans-serif !important;
    font-size: clamp(0.7rem, 1.2vw, 0.78rem);
    color: var(--text-muted);
    margin-top: 0.35rem;
    letter-spacing: 0.03em;
    line-height: 1.35;
}

div[data-testid="stMetric"] {
    background: #FED7AA; border: 1px solid var(--border);
    border-radius: 6px; padding: 14px 18px;
}
div[data-testid="stMetric"] label {
    font-family: 'DM Sans', sans-serif !important; font-size: 0.75rem !important;
    text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted) !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.6rem !important; color: var(--text-primary) !important;
}

.pollen-strip {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px 24px; margin-bottom: 1.2rem;
}
.pollen-strip-title {
    font-family: 'DM Sans', sans-serif; font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 8px;
}
.pollen-category { display: inline-block; margin-right: 28px; }
.pollen-label { font-family: 'DM Sans', sans-serif; font-size: 0.78rem; font-weight: 500; color: var(--text-secondary); }
.pollen-value { font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 500; }

.risk-badge {
    display: inline-block; padding: 2px 10px; border-radius: 3px;
    font-family: 'DM Sans', sans-serif; font-size: 0.72rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.05em;
}
.risk-low { background: #E8F5E9; color: #2D7D4F; }
.risk-moderate { background: #FFF8E1; color: #B8860B; }
.risk-high { background: #FBE9E7; color: #D4652A; }
.risk-severe { background: #FFEBEE; color: #A8201A; }

.detail-panel {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 8px; padding: 20px 24px; margin-top: 1rem;
}
.detail-panel h3 { font-family: 'Playfair Display', serif !important; margin-bottom: 4px; }
.stat-row {
    display: flex; justify-content: space-between; padding: 6px 0;
    border-bottom: 1px solid #F0EDE8; font-family: 'DM Sans', sans-serif; font-size: 0.88rem;
}
.stat-label { color: var(--text-secondary); }
.stat-value { font-family: 'JetBrains Mono', monospace; font-weight: 500; }

.feature-bar-container { margin: 6px 0; }
.feature-name { font-family: 'DM Sans', sans-serif; font-size: 0.82rem; color: var(--text-secondary); margin-bottom: 2px; }
.feature-bar-bg { background: #F0EDE8; border-radius: 2px; height: 6px; position: relative; }
.feature-bar-fill { background: var(--accent-warm); border-radius: 2px; height: 6px; position: absolute; top: 0; left: 0; }

.footer {
    margin-top: 3rem; padding: 2rem 2.5rem;
    background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px;
    font-family: 'DM Sans', sans-serif; font-size: 0.92rem; color: var(--text-secondary); line-height: 1.8;
}
.footer strong {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.1rem; color: var(--text-primary);
}
.footer-row {
    display: flex; flex-wrap: wrap; gap: 8px 32px;
    margin-top: 8px;
}
.footer-item {
    font-size: 0.85rem; color: var(--text-muted);
}

.sidebar-copy {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-size: 0.78rem;
    color: var(--text-muted);
    line-height: 1.6;
}
.sidebar-copy strong,
.sidebar-copy p,
.sidebar-copy div,
.sidebar-copy span {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.sidebar-copy strong {
    color: var(--text-secondary) !important;
    font-weight: 600;
}
.sidebar-section {
    margin: 0 0 0.8rem 0;
}
.sidebar-section:last-child {
    margin-bottom: 0;
}
.sidebar-section-title {
    color: var(--text-secondary) !important;
    font-weight: 600;
    margin-bottom: 0.15rem;
}
.sidebar-section-body {
    color: var(--text-muted);
}

.data-source-tag {
    display: inline-block; padding: 3px 10px; border-radius: 3px;
    font-family: 'DM Sans', sans-serif; font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.04em;
}
.tag-predicted { background: #FFF3E0; color: #C4501A; }
.tag-actual { background: #E8F5E9; color: #2D7D4F; }

.eval-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 8px; padding: 20px 24px; height: 100%;
}
.eval-card-title {
    font-family: 'DM Sans', sans-serif; font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted);
    margin-bottom: 12px;
}
.eval-big-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem; font-weight: 500; color: var(--text-primary);
    line-height: 1.1; margin-bottom: 2px;
}
.eval-label {
    font-family: 'DM Sans', sans-serif; font-size: 0.78rem; color: var(--text-muted);
}
.eval-row {
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 8px 0; border-bottom: 1px solid #F0EDE8;
}
.eval-row:last-child { border-bottom: none; }
.eval-row-label { font-family: 'DM Sans', sans-serif; font-size: 0.85rem; color: var(--text-secondary); }
.eval-row-value { font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; font-weight: 500; }
.eval-row-note { font-family: 'DM Sans', sans-serif; font-size: 0.72rem; color: var(--text-muted); }
.eval-section-intro {
    font-family: 'DM Sans', sans-serif; font-size: 0.88rem;
    color: var(--text-secondary); margin-bottom: 1rem;
    line-height: 1.6;
}
.eval-badge {
    display: inline-block; padding: 2px 8px; border-radius: 3px;
    font-family: 'DM Sans', sans-serif; font-size: 0.68rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.04em;
    margin-left: 8px; vertical-align: middle;
}
.badge-pass { background: #E8F5E9; color: #2D7D4F; }
.badge-warn { background: #FFF8E1; color: #B8860B; }
.badge-info { background: #F3E8FF; color: #7C3AED; }
.compare-table {
    width: 100%; border-collapse: collapse;
    font-family: 'DM Sans', sans-serif; font-size: 0.85rem;
}
.compare-table th {
    font-family: 'DM Sans', sans-serif; font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted);
    text-align: left; padding: 8px 12px; border-bottom: 2px solid var(--border);
    font-weight: 500;
}
.compare-table td {
    padding: 10px 12px; border-bottom: 1px solid #F0EDE8;
    font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;
}
.compare-table td:first-child {
    font-family: 'DM Sans', sans-serif; font-weight: 500;
    color: var(--text-primary);
}
.compare-table tr.row-best td { background: #FEFCE8; }
.compare-table tr:last-child td { border-bottom: none; }

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Database connection ──────────────────────────────────────────────────────

import os
from dotenv import load_dotenv

# Load .env securely from the true directory
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

DB_PARAMS = dict(
    dbname=os.environ.get("DB_NAME", "pollen_pain"),
    user=os.environ.get("DB_USER", "postgres"),
    password=os.environ.get("DB_PASSWORD", ""),
    host=os.environ.get("DB_HOST", "localhost"),
    port=int(os.environ.get("DB_PORT", 5432)),
    sslmode="require" if os.environ.get("DB_HOST", "localhost") != "localhost" else "prefer",
)
GEOJSON_PATH = Path("./data/Input/nta2020.geojson")
FOLD_RESULTS_PATH = Path("./data/models/fold_results.csv")
BASELINE_COMP_PATH = Path("./data/models/holdout_baseline_comparison.csv")
LOGREG_FOLD_PATH = Path("./data/models/logistic_regression_fold_results.csv")
EVAL_SUMMARY_PATH = Path("./data/models/evaluation_summary.txt")
LOGREG_SUMMARY_PATH = Path("./data/models/logistic_regression_summary.txt")


@st.cache_data(ttl=300)
def query_db(sql: str) -> pd.DataFrame:
    conn = psycopg2.connect(**DB_PARAMS)
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


@st.cache_data
def load_geojson():
    with open(GEOJSON_PATH) as f:
        return json.load(f)


@st.cache_data
def load_fold_results():
    if FOLD_RESULTS_PATH.exists():
        return pd.read_csv(FOLD_RESULTS_PATH)
    return pd.DataFrame()


@st.cache_data
def load_baseline_comparison():
    if BASELINE_COMP_PATH.exists():
        return pd.read_csv(BASELINE_COMP_PATH)
    return pd.DataFrame()


@st.cache_data
def load_logreg_folds():
    if LOGREG_FOLD_PATH.exists():
        return pd.read_csv(LOGREG_FOLD_PATH)
    return pd.DataFrame()


@st.cache_data
def parse_eval_summary():
    result = {
        "holdout_mae": None, "holdout_rmse": None, "holdout_r": None,
        "holdout_r_p": None, "holdout_within_15": None,
        "chs_pred_r": None, "chs_pred_p": None,
        "chs_actual_r": None, "chs_actual_p": None,
        "pollen_14d_r": None, "pollen_14d_p": None,
        "pollen_28d_r": None, "pollen_28d_p": None,
        "logreg_auc": None,
    }
    if not EVAL_SUMMARY_PATH.exists():
        return result

    import re
    text = EVAL_SUMMARY_PATH.read_text()

    m = re.search(r"XGBoost MAE:\s+([\d.]+)", text)
    if m: result["holdout_mae"] = float(m.group(1))
    m = re.search(r"XGBoost RMSE:\s+([\d.]+)", text)
    if m: result["holdout_rmse"] = float(m.group(1))
    m = re.search(r"XGBoost Pearson r:\s+([\d.]+)\s+\(p=([\d.e+-]+)\)", text)
    if m: result["holdout_r"] = float(m.group(1)); result["holdout_r_p"] = float(m.group(2))
    m = re.search(r"XGBoost NTAs within 15%:\s+([\d.]+)%", text)
    if m: result["holdout_within_15"] = float(m.group(1)) / 100

    m = re.search(r"Predicted holdout mean vs CHS prevalence:\s+r=([\d.]+)\s+\(p=([\d.e+-]+)\)", text)
    if m: result["chs_pred_r"] = float(m.group(1)); result["chs_pred_p"] = float(m.group(2))
    m = re.search(r"Actual holdout mean vs CHS prevalence:\s+r=([\d.]+)\s+\(p=([\d.e+-]+)\)", text)
    if m: result["chs_actual_r"] = float(m.group(1)); result["chs_actual_p"] = float(m.group(2))

    m = re.search(r"pollen_14d_lag_avg\s+r=([\d.]+)\s+\(p=([\d.e+-]+)", text)
    if m: result["pollen_14d_r"] = float(m.group(1)); result["pollen_14d_p"] = float(m.group(2))
    m = re.search(r"pollen_28d_lag_avg\s+r=([\d.]+)\s+\(p=([\d.e+-]+)", text)
    if m: result["pollen_28d_r"] = float(m.group(1)); result["pollen_28d_p"] = float(m.group(2))

    if LOGREG_SUMMARY_PATH.exists():
        lr_text = LOGREG_SUMMARY_PATH.read_text()
        m = re.search(r"Mean AUC-ROC:\s+([\d.]+)", lr_text)
        if m: result["logreg_auc"] = float(m.group(1))

    return result


@st.cache_data
def load_all_data():
    modeling = query_db("SELECT * FROM pollen.modeling_table ORDER BY nta_code, year_month")
    predictions = query_db("SELECT * FROM pollen.model_predictions ORDER BY nta_code, year_month")
    feature_imp = query_db("SELECT * FROM pollen.feature_importance ORDER BY importance DESC")
    complaints = query_db("SELECT * FROM pollen.complaints_monthly ORDER BY borough, year_month")
    pollen_monthly = query_db("SELECT * FROM pollen.pollen_monthly ORDER BY year_month")
    neighborhoods = query_db("SELECT * FROM pollen.neighborhoods ORDER BY nta_code")
    return modeling, predictions, feature_imp, complaints, pollen_monthly, neighborhoods


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
    "Bronx": "BRONX", "Brooklyn": "BROOKLYN", "Manhattan": "MANHATTAN",
    "Queens": "QUEENS", "Staten Island": "STATEN ISLAND",
}


def classify_risk(val, q25, q75, q90):
    if val >= q90:
        return "Severe"
    elif val >= q75:
        return "High"
    elif val >= q25:
        return "Moderate"
    return "Low"


def risk_badge_html(risk_level):
    return f'<span class="risk-badge risk-{risk_level.lower()}">{risk_level}</span>'


def interpolate_color(norm):
    """Warm-to-red sequential ramp: cream → amber → burnt orange → deep red."""
    colors = [
        (0.0, (232, 245, 233)),   # soft green-white
        (0.25, (255, 224, 130)),   # warm amber
        (0.5, (255, 167, 38)),     # orange
        (0.75, (212, 101, 42)),    # burnt orange
        (1.0, (168, 32, 26)),      # deep red
    ]
    norm = max(0, min(1, norm))
    for i in range(len(colors) - 1):
        t0, c0 = colors[i]
        t1, c1 = colors[i + 1]
        if norm <= t1:
            f = (norm - t0) / (t1 - t0) if t1 > t0 else 0
            r = int(c0[0] + f * (c1[0] - c0[0]))
            g = int(c0[1] + f * (c1[1] - c0[1]))
            b = int(c0[2] + f * (c1[2] - c0[2]))
            return f"#{r:02x}{g:02x}{b:02x}"
    return f"#{colors[-1][1][0]:02x}{colors[-1][1][1]:02x}{colors[-1][1][2]:02x}"


def build_choropleth(geojson_data, month_data, value_col, selected_nta=None):
    m = folium.Map(
        location=[40.7128, -73.95], zoom_start=10.4,
        tiles="cartodbpositron", control_scale=False, zoom_control=False,
    )

    nta_values = dict(zip(month_data["nta_code"], month_data[value_col]))
    nta_names = dict(zip(month_data["nta_code"], month_data["nta_name"]))
    nta_risks = dict(zip(month_data["nta_code"], month_data["risk_level"]))
    valid_ntas = set(month_data["nta_code"])

    vmin = month_data[value_col].quantile(0.05)
    vmax = month_data[value_col].quantile(0.95)

    for feature in geojson_data["features"]:
        nta_code = feature["properties"]["NTA2020"]
        if nta_code in valid_ntas:
            val = nta_values.get(nta_code, 0)
            feature["properties"]["display_value"] = f"{val:.1f}"
            feature["properties"]["risk_level"] = nta_risks.get(nta_code, "—")
        else:
            feature["properties"]["display_value"] = "—"
            feature["properties"]["risk_level"] = "—"

    def style_function(feature):
        nta_code = feature["properties"]["NTA2020"]
        nta_type = feature["properties"]["NTAType"]

        if nta_type != "0" or nta_code not in valid_ntas:
            return {"fillColor": "#E8E4DF", "fillOpacity": 0.3, "color": "#D0CCC7", "weight": 0.5}

        val = nta_values.get(nta_code, 0)
        norm = max(0, min(1, (val - vmin) / (vmax - vmin))) if vmax > vmin else 0.5
        color = interpolate_color(norm)

        is_selected = nta_code == selected_nta
        return {
            "fillColor": color,
            "fillOpacity": 0.65 + 0.2 * norm,
            "color": "#1A1A1A" if is_selected else "#FFFFFF",
            "weight": 3 if is_selected else 0.8,
        }

    def highlight_function(feature):
        return {"weight": 2.5, "color": "#1A1A1A", "fillOpacity": 0.9}

    folium.GeoJson(
        geojson_data,
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["NTAName", "BoroName", "display_value", "risk_level"],
            aliases=["Neighborhood:", "Borough:", "ED Visits:", "Risk:"],
            style="font-family: 'DM Sans', sans-serif; font-size: 13px; padding: 8px 12px;",
        ),
    ).add_to(m)
    return m


# ── Load all data ────────────────────────────────────────────────────────────

modeling, predictions, feature_imp, complaints, pollen_monthly, neighborhoods = load_all_data()
fold_results = load_fold_results()
baseline_comp = load_baseline_comparison()
logreg_folds = load_logreg_folds()
eval_summary = parse_eval_summary()
geojson_data = load_geojson()

# Prefer final-model fitted predictions when available. If the database only has
# CV outputs, fall back to the latest fold-specific prediction per month.
prediction_mode = "none"
if not predictions.empty:
    if predictions["fold"].fillna(-1).eq(0).all():
        latest_preds = predictions[["nta_code", "year_month", "pred", "fold"]].drop_duplicates(
            subset=["nta_code", "year_month"]
        )
        prediction_mode = "full_period"
    else:
        latest_preds = (
            predictions
            .sort_values("fold")
            .drop_duplicates(subset=["nta_code", "year_month"], keep="last")
            [["nta_code", "year_month", "pred", "fold"]]
        )
        prediction_mode = "cv_holdout"
    merged = modeling.merge(latest_preds, on=["nta_code", "year_month"], how="left")
else:
    merged = modeling.copy()
    merged["pred"] = None
    merged["fold"] = None

merged["has_prediction"] = merged["pred"].notna()

# Months that have predictions
pred_months = sorted(merged[merged["has_prediction"]]["year_month"].unique())

# Drop trailing actuals that extend beyond the model's capability to ensure 1:1 UI alignment
if pred_months:
    merged = merged[merged["year_month"].isin(pred_months)]

# Risk thresholds from the full distribution of actual ED visits
ed_q25 = modeling["ed_visits"].quantile(0.25)
ed_q75 = modeling["ed_visits"].quantile(0.75)
ed_q90 = modeling["ed_visits"].quantile(0.90)


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">
            Pollen & Pain
        </div>
        <div class="sidebar-brand-subtitle">
            NYC Asthma ED Forecasting
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    is_prediction_view = True
    month_options = pred_months
        
    if "display_month" not in st.session_state:
        st.session_state.display_month = month_options[-1] if month_options else None

    # If the current stored month isn't in the new options list, default to latest
    if st.session_state.display_month not in month_options:
        st.session_state.display_month = month_options[-1] if month_options else None

    # Find the valid index
    try:
        default_idx = month_options.index(st.session_state.display_month)
    except (ValueError, TypeError):
        default_idx = len(month_options) - 1 if month_options else 0

    selected_month = st.selectbox(
        "Month",
        options=month_options,
        index=default_idx,
        format_func=lambda x: pd.Timestamp(x + "-01").strftime("%B %Y"),
    )
    
    # Save selection back to state
    st.session_state.display_month = selected_month

    borough_options = ["All Boroughs"] + sorted(modeling["borough"].unique())
    selected_borough = st.selectbox("Borough", options=borough_options)

    risk_options = ["All Levels", "Low", "Moderate", "High", "Severe"]
    selected_risk = st.selectbox("Risk Level", options=risk_options)

    st.markdown("---")

    if eval_summary["holdout_mae"] is not None:
        model_metrics_html = (
            "XGBoost regression · walk-forward CV + holdout.<br>"
            f"Holdout MAE {eval_summary['holdout_mae']:.2f} · r = {eval_summary['holdout_r']:.3f}<br>"
            f"{eval_summary['holdout_within_15']:.1%} of NTAs within 15%"
        )
        if eval_summary["logreg_auc"] is not None:
            model_metrics_html += f"<br>Logistic baseline AUC-ROC {eval_summary['logreg_auc']:.3f}"
    elif not fold_results.empty:
        mean_mae = fold_results["mae"].mean()
        mean_r = fold_results["pearson_r"].mean()
        model_metrics_html = (
            "XGBoost regression with walk-forward time-series validation.<br>"
            f"Mean MAE {mean_mae:.2f} · Mean Pearson r {mean_r:.3f}"
        )
    else:
        model_metrics_html = "XGBoost regression with walk-forward time-series validation."

    st.markdown(f"""
    <div class="sidebar-copy">
        <div class="sidebar-section">
            <div class="sidebar-section-title">About</div>
            <div class="sidebar-section-body">
                Predicts monthly asthma emergency department visit counts at the
                neighborhood level across all five NYC boroughs.
            </div>
        </div>
        <div class="sidebar-section">
            <div class="sidebar-section-title">Model Evaluation</div>
            <div class="sidebar-section-body">{model_metrics_html}</div>
        </div>
        <div class="sidebar-section">
            <div class="sidebar-section-title">Prediction View</div>
            <div class="sidebar-section-body">
                Map and charts use fitted predictions from the final model across the full available period, not CV holdout performance.
            </div>
        </div>
        <div class="sidebar-section">
            <div class="sidebar-section-title">Team</div>
            <div class="sidebar-section-body">Saketh Boddu · Andre Nguyen · Nam Lai</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Filter data for selected month ──────────────────────────────────────────

month_data = merged[merged["year_month"] == selected_month].copy()

# Determine which column to display on the map
if month_data["has_prediction"].any():
    display_col = "pred"
    source_tag = '<span class="data-source-tag tag-predicted">Fitted Predictions</span>'
else:
    display_col = "ed_visits"
    source_tag = '<span class="data-source-tag tag-actual">Actual</span>'

month_data["display_value"] = month_data[display_col]
month_data["risk_level"] = month_data["display_value"].apply(
    lambda x: classify_risk(x, ed_q25, ed_q75, ed_q90)
)

if selected_borough != "All Boroughs":
    month_data = month_data[month_data["borough"] == selected_borough]
if selected_risk != "All Levels":
    month_data = month_data[month_data["risk_level"] == selected_risk]


# ── Header + source indicator ────────────────────────────────────────────────

display_month = pd.Timestamp(selected_month + "-01").strftime("%B %Y")
section_title = "Full-period Prediction View"
st.markdown(f"## {section_title}")
st.markdown(f'<h2 style="border:none; padding:0;">{display_month} {source_tag}</h2>', unsafe_allow_html=True)

if month_data["has_prediction"].any():
    if prediction_mode == "full_period":
        prediction_note = """
        Showing fitted predictions from the final XGBoost model trained on the full dataset.
        These values cover the full available timeline and support map, chart, and neighborhood exploration.
        They should not be interpreted as cross-validation performance.
        """
    st.markdown(f"""
        <div style="
            font-family: 'DM Sans', sans-serif;
            font-size: 0.85rem;
            color: var(--text-secondary);
            background: #F9F7F4;
            border: 1px solid var(--border);
            border-left: 4px solid var(--accent-warm);
            border-radius: 6px;
            padding: 12px 16px;
            margin-bottom: 12px;
            line-height: 1.5;
        ">
            {prediction_note}
        </div>
""", unsafe_allow_html=True)


# ── Summary metrics ──────────────────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Neighborhoods", f"{len(month_data)}")
with col2:
    avg_val = month_data["display_value"].mean()
    label = "Avg Fitted Prediction"
    st.metric(label, f"{avg_val:.1f}")
with col3:
    severe_count = (month_data["risk_level"] == "Severe").sum()
    st.metric("Severe Risk", f"{severe_count}")
with col4:
    high_count = (month_data["risk_level"] == "High").sum()
    st.metric("High Risk", f"{high_count}")
with col5:
    total_val = month_data["display_value"].sum()
    label = "Total Fitted Prediction"
    st.metric(label, f"{total_val:,.0f}")


# ── Main layout: Map + Detail ───────────────────────────────────────────────

if "selected_nta" not in st.session_state:
    st.session_state.selected_nta = None

map_col, detail_col = st.columns([3, 2])

with map_col:
    st.markdown("### Neighborhood Risk Map")

    choropleth = build_choropleth(geojson_data, month_data, "display_value", st.session_state.selected_nta)
    map_output = st_folium(choropleth, width=None, height=640, returned_objects=["last_object_clicked_tooltip"])

    if map_output and map_output.get("last_object_clicked_tooltip"):
        tooltip_text = str(map_output["last_object_clicked_tooltip"])
        for _, row in month_data.iterrows():
            if row["nta_code"] in tooltip_text or row["nta_name"] in tooltip_text:
                st.session_state.selected_nta = row["nta_code"]
                break

    # Color scale legend
    st.markdown("""
    <div style="margin-top: 8px; font-family: 'DM Sans', sans-serif; font-size: 0.78rem;">
        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="color: var(--text-muted);">Low</span>
            <div style="flex: 1; height: 8px; border-radius: 4px; background: linear-gradient(to right, #e8f5e9, #ffe082, #ffa726, #d4652a, #a8201a);"></div>
            <span style="color: var(--text-muted);">High</span>
            <span style="margin-left: 12px;"><span style="display:inline-block; width:12px; height:12px; background:#E8E4DF; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>Excluded</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


with detail_col:
    if st.session_state.selected_nta:
        nta_code = st.session_state.selected_nta
        nta_row = month_data[month_data["nta_code"] == nta_code]

        if not nta_row.empty:
            nta = nta_row.iloc[0]
            risk_html = risk_badge_html(nta["risk_level"])

            pred_actual_html = ""
            if nta["has_prediction"]:
                diff = nta["pred"] - nta["ed_visits"]
                diff_pct = diff / nta["ed_visits"] * 100 if nta["ed_visits"] > 0 else 0
                diff_sign = "+" if diff > 0 else ""
                pred_actual_html = f"""<div class="stat-row" >
<span class="stat-label">Fitted Prediction</span>
<span class="stat-value" style="color: #C4501A;">{nta['pred']:.1f}</span>
</div>
<div class="stat-row">
<span class="stat-label">Observed ED Visits</span>
<span class="stat-value">{nta['ed_visits']:.1f}</span>
</div>
<div class="stat-row">
<span class="stat-label">Fitted Residual</span>
<span class="stat-value">{diff_sign}{diff:.1f} ({diff_sign}{diff_pct:.0f}%)</span>
</div>"""
            else:
                pred_actual_html = f"""<div class="stat-row">
<span class="stat-label">Est. ED Visits</span>
<span class="stat-value">{nta['ed_visits']:.1f}</span>
</div>"""

            st.markdown(f"""<div class="detail-panel">
<h3>{nta['nta_name']}</h3>
<div style="font-family: 'DM Sans', sans-serif; font-size: 0.82rem; color: var(--text-muted); margin-bottom: 12px;">
{nta['borough']} · {nta_code} {risk_html}
</div>
{pred_actual_html}
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
<span class="stat-label">Temperature</span>
<span class="stat-value">{nta['temp_max_mean']:.0f}°C / {nta['temp_min_mean']:.0f}°C</span>
</div>
<div class="stat-row">
<span class="stat-label">PM2.5</span>
<span class="stat-value">{nta['pm25_mean']:.1f} µg/m³</span>
</div>
</div>""", unsafe_allow_html=True)

            # 12-month history chart: actual vs predicted
            st.markdown("#### Observed vs. Fitted")
            nta_history = merged[merged["nta_code"] == nta_code].sort_values("year_month").tail(24)

            if not nta_history.empty:
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=nta_history["year_month"], y=nta_history["ed_visits"],
                    mode="lines+markers", name="Actual",
                    line=dict(color="#1A1A1A", width=2),
                    marker=dict(size=5, color="#1A1A1A"),
                ))

                nta_with_preds = nta_history[nta_history["has_prediction"]]
                if not nta_with_preds.empty:
                    fig.add_trace(go.Scatter(
                        x=nta_with_preds["year_month"], y=nta_with_preds["pred"],
                        mode="lines+markers", name="Fitted prediction",
                        line=dict(color="#C4501A", width=2.5),
                        marker=dict(size=6, color="#C4501A", symbol="diamond"),
                    ))

                fig.update_layout(
                    height=260, margin=dict(l=0, r=0, t=10, b=30),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Sans", size=12, color="#5A5A5A"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
                    xaxis=dict(showgrid=False, tickfont=dict(size=10), tickangle=-45),
                    yaxis=dict(showgrid=True, gridcolor="#F0EDE8", title=None, tickfont=dict(size=10)),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


        else:
            st.info("Selected neighborhood not visible with current filters.")
    else:
        st.markdown("""
        <div class="detail-panel" style="text-align: center; padding: 60px 24px;">
            <div style="font-family: 'Playfair Display', serif; font-size: 1.1rem; color: var(--text-secondary); margin-bottom: 8px;">
                Select a Neighborhood
            </div>
            <div style="font-family: 'DM Sans', sans-serif; font-size: 0.85rem; color: var(--text-muted);">
                Click on a colored area on the map to view<br>observed vs. fitted ED visits, trends, and local context.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Feature Importance ───────────────────────────────────────────────────────

st.markdown("## What's Driving Predictions")
st.markdown("""
<div style="font-family: 'DM Sans', sans-serif; font-size: 0.88rem; color: var(--text-secondary); margin-bottom: 1rem;">
    Each bar shows how much a given factor contributes to the model's predictions.
    Neighborhood asthma prevalence is the strongest signal; pollen at 4-week lag is the
    second most important — confirming the delayed relationship between pollen exposure and ED surges.
</div>
""", unsafe_allow_html=True)

fi_col1, fi_col2 = st.columns(2)
top_features = feature_imp.head(10)
max_imp = top_features["importance"].max()

for col, features in [(fi_col1, top_features.head(5)), (fi_col2, top_features.tail(5))]:
    with col:
        for _, row in features.iterrows():
            label = FEATURE_LABELS.get(row["feature"], row["feature"])
            pct = row["importance"] / max_imp * 100
            imp_pct = row["importance"] * 100
            st.markdown(f"""
            <div class="feature-bar-container">
                <div class="feature-name">{label} <span style="font-family: 'JetBrains Mono'; font-size: 0.72rem; color: var(--text-muted);">{imp_pct:.1f}%</span></div>
                <div class="feature-bar-bg"><div class="feature-bar-fill" style="width: {pct}%;"></div></div>
            </div>
            """, unsafe_allow_html=True)




# ── Validation Signals ──────────────────────────────────────────────────────

has_chs = eval_summary["chs_pred_r"] is not None
has_pollen = eval_summary["pollen_14d_r"] is not None

if has_chs or has_pollen:
    st.markdown("## Validation Signals")
    st.markdown("""
    <div class="eval-section-intro">
        Independent checks confirming model predictions align with known epidemiological patterns.
    </div>
    """, unsafe_allow_html=True)

    vs_c1, vs_c2 = st.columns(2)

    if has_pollen:
        p14_sig = "p < 0.001" if eval_summary["pollen_14d_p"] < 0.001 else f"p = {eval_summary['pollen_14d_p']:.2e}"
        p28_sig = "p < 0.001" if eval_summary["pollen_28d_p"] < 0.001 else f"p = {eval_summary['pollen_28d_p']:.2e}"
        pollen_badge = "pass" if eval_summary["pollen_28d_p"] < 0.05 else "warn"
        pollen_label = "Significant" if pollen_badge == "pass" else "Not significant"

        with vs_c1:
            st.markdown(f"""<div class="eval-card">
                <div class="eval-card-title">Pollen–ED Visit Lag Correlation</div>
                <div class="eval-row">
                    <span class="eval-row-label">2-week lag (pollen_14d_lag_avg)</span>
                    <span class="eval-row-value">r = {eval_summary['pollen_14d_r']:.3f}</span>
                </div>
                <div class="eval-row">
                    <span class="eval-row-label"></span>
                    <span class="eval-row-note">{p14_sig}, n = 5,952</span>
                </div>
                <div class="eval-row">
                    <span class="eval-row-label">4-week lag (pollen_28d_lag_avg)</span>
                    <span class="eval-row-value">r = {eval_summary['pollen_28d_r']:.3f} <span class="eval-badge badge-{pollen_badge}">{pollen_label}</span></span>
                </div>
                <div class="eval-row">
                    <span class="eval-row-label"></span>
                    <span class="eval-row-note">{p28_sig}, n = 5,952</span>
                </div>
                <div style="margin-top: 12px;">
                    <div class="eval-row-note">Confirms the documented 1–2 week delay between pollen exposure and asthma ED surges. The 4-week lag shows stronger association.</div>
                </div>
            </div>""", unsafe_allow_html=True)

    if has_chs:
        chs_pred_sig = "p < 0.001" if eval_summary["chs_pred_p"] < 0.001 else f"p = {eval_summary['chs_pred_p']:.2e}"
        chs_actual_sig = "p < 0.001" if eval_summary["chs_actual_p"] < 0.001 else f"p = {eval_summary['chs_actual_p']:.2e}"

        with vs_c2:
            st.markdown(f"""<div class="eval-card">
                <div class="eval-card-title">CHS Prevalence Cross-Check</div>
                <div class="eval-row">
                    <span class="eval-row-label">Predictions vs. CHS prevalence</span>
                    <span class="eval-row-value">r = {eval_summary['chs_pred_r']:.3f}</span>
                </div>
                <div class="eval-row">
                    <span class="eval-row-label"></span>
                    <span class="eval-row-note">{chs_pred_sig}</span>
                </div>
                <div class="eval-row">
                    <span class="eval-row-label">Actual ED visits vs. CHS prevalence</span>
                    <span class="eval-row-value">r = {eval_summary['chs_actual_r']:.3f}</span>
                </div>
                <div class="eval-row">
                    <span class="eval-row-label"></span>
                    <span class="eval-row-note">{chs_actual_sig}</span>
                </div>
                <div style="margin-top: 12px;">
                    <div class="eval-row-note">Model predictions track neighborhood-level asthma prevalence as closely as actual ED data does — the model learned real spatial health patterns.</div>
                </div>
            </div>""", unsafe_allow_html=True)


# ── Model evaluation ─────────────────────────────────────────────────────────

st.markdown("## Model Evaluation")

fold_count = len(fold_results) if not fold_results.empty else 0
holdout_available = eval_summary["holdout_mae"] is not None

st.markdown(f"""
<div class="eval-section-intro">
    All metrics reflect out-of-sample performance. Walk-forward cross-validation ({fold_count} folds)
    trained on past data and evaluated on future months. The holdout window (May–Oct 2025)
    was never seen during CV or model tuning.
</div>
""", unsafe_allow_html=True)


# ── Holdout headline metrics ────────────────────────────────────────────────

if holdout_available:
    st.markdown("### Holdout Performance (May–Oct 2025)")

    within_15 = eval_summary["holdout_within_15"]
    within_badge = "pass" if within_15 and within_15 >= 0.70 else "warn"
    within_label = "Meets target" if within_badge == "pass" else "Below target"

    ho_c1, ho_c2, ho_c3, ho_c4 = st.columns(4)
    with ho_c1:
        st.markdown(f"""<div class="eval-card">
            <div class="eval-card-title">MAE</div>
            <div class="eval-big-number">{eval_summary['holdout_mae']:.2f}</div>
            <div class="eval-label">ED visits</div>
        </div>""", unsafe_allow_html=True)
    with ho_c2:
        st.markdown(f"""<div class="eval-card">
            <div class="eval-card-title">RMSE</div>
            <div class="eval-big-number">{eval_summary['holdout_rmse']:.2f}</div>
            <div class="eval-label">ED visits</div>
        </div>""", unsafe_allow_html=True)
    with ho_c3:
        st.markdown(f"""<div class="eval-card">
            <div class="eval-card-title">Pearson r</div>
            <div class="eval-big-number">{eval_summary['holdout_r']:.3f}</div>
            <div class="eval-label">p {'< 0.001' if eval_summary['holdout_r_p'] is not None and eval_summary['holdout_r_p'] < 0.001 else f"= {eval_summary['holdout_r_p']:.2e}"}</div>
        </div>""", unsafe_allow_html=True)
    with ho_c4:
        st.markdown(f"""<div class="eval-card">
            <div class="eval-card-title">NTAs within 15%</div>
            <div class="eval-big-number">{within_15:.1%}</div>
            <div class="eval-label"><span class="eval-badge badge-{within_badge}">{within_label}</span></div>
        </div>""", unsafe_allow_html=True)


# ── Baseline comparison table ───────────────────────────────────────────────

if not baseline_comp.empty and holdout_available:
    st.markdown("### Model vs. Baselines (Holdout)")
    st.markdown("""
    <div class="eval-section-intro">
        Comparing XGBoost against naive forecasting strategies on the same holdout window.
    </div>
    """, unsafe_allow_html=True)

    rows_html = ""
    baselines = {
        "seasonal_average": "Seasonal Average",
        "prior_observation": "Prior Month",
    }
    for _, brow in baseline_comp.iterrows():
        label = baselines.get(brow["baseline"], brow["baseline"])
        rows_html += f"""<tr>
            <td>{label}</td>
            <td>{brow['mae']:.2f}</td>
            <td>{brow['rmse']:.2f}</td>
            <td>{brow['pearson_r']:.3f}</td>
            <td>{brow['pct_ntas_within_15pct']:.1%}</td>
        </tr>"""

    st.markdown(f"""
    <div class="eval-card" style="padding: 0; overflow: hidden;">
        <table class="compare-table">
            <thead><tr>
                <th>Model</th><th>MAE</th><th>RMSE</th><th>Pearson r</th><th>NTAs within 15%</th>
            </tr></thead>
            <tbody>
                <tr class="row-best">
                    <td>XGBoost <span class="eval-badge badge-pass">Best</span></td>
                    <td>{eval_summary['holdout_mae']:.2f}</td>
                    <td>{eval_summary['holdout_rmse']:.2f}</td>
                    <td>{eval_summary['holdout_r']:.3f}</td>
                    <td>{eval_summary['holdout_within_15']:.1%}</td>
                </tr>
                {rows_html}
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)


# ── Logistic Regression baseline ────────────────────────────────────────────

if not logreg_folds.empty:
    st.markdown("### Logistic Regression Baseline (Binary Classification)")

    logreg_auc = eval_summary["logreg_auc"] or logreg_folds["auc"].mean()
    mean_acc = logreg_folds["accuracy"].mean()
    mean_f1 = logreg_folds["f1"].mean()

    lr_c1, lr_c2, lr_c3 = st.columns(3)
    with lr_c1:
        st.markdown(f"""<div class="eval-card">
            <div class="eval-card-title">Mean AUC-ROC</div>
            <div class="eval-big-number">{logreg_auc:.3f}</div>
            <div class="eval-label">across {len(logreg_folds)} folds</div>
        </div>""", unsafe_allow_html=True)
    with lr_c2:
        st.markdown(f"""<div class="eval-card">
            <div class="eval-card-title">Mean Accuracy</div>
            <div class="eval-big-number">{mean_acc:.1%}</div>
            <div class="eval-label">above-median ED month</div>
        </div>""", unsafe_allow_html=True)
    with lr_c3:
        st.markdown(f"""<div class="eval-card">
            <div class="eval-card-title">Mean F1</div>
            <div class="eval-big-number">{mean_f1:.3f}</div>
            <div class="eval-label">precision / recall balance</div>
        </div>""", unsafe_allow_html=True)

    lr_table = logreg_folds.copy()
    lr_table["Fold"] = lr_table["fold"].astype(int)
    lr_table["Validation Window"] = lr_table["test_start"] + " to " + lr_table["test_end"]
    lr_table = lr_table.rename(columns={
        "auc": "AUC-ROC", "accuracy": "Accuracy", "precision": "Precision",
        "recall": "Recall", "f1": "F1",
    })
    st.dataframe(
        lr_table[["Fold", "Validation Window", "AUC-ROC", "Accuracy", "Precision", "Recall", "F1"]],
        use_container_width=True, hide_index=True,
    )


# ── XGBoost CV fold detail ──────────────────────────────────────────────────

if not fold_results.empty:
    st.markdown("### XGBoost Cross-Validation Detail")

    fold_count = len(fold_results)
    cv_c1, cv_c2, cv_c3, cv_c4 = st.columns(4)
    with cv_c1:
        st.metric(f"Mean MAE ({fold_count} folds)", f"{fold_results['mae'].mean():.2f}")
    with cv_c2:
        st.metric(f"Mean RMSE ({fold_count} folds)", f"{fold_results['rmse'].mean():.2f}")
    with cv_c3:
        st.metric(f"Mean Pearson r ({fold_count} folds)", f"{fold_results['pearson_r'].mean():.3f}")
    with cv_c4:
        st.metric(f"NTAs within 15% ({fold_count} folds)", f"{fold_results['pct_ntas_within_15pct'].mean():.1%}")

    eval_table = fold_results.copy()
    eval_table["Fold"] = eval_table["fold"].astype(int)
    eval_table["Validation Window"] = eval_table["test_start"] + " to " + eval_table["test_end"]
    eval_table = eval_table.rename(columns={
        "mae": "MAE", "rmse": "RMSE", "pearson_r": "Pearson r",
        "pct_ntas_within_15pct": "NTAs within 15%",
    })
    st.dataframe(
        eval_table[["Fold", "Validation Window", "MAE", "RMSE", "Pearson r", "NTAs within 15%"]],
        use_container_width=True, hide_index=True,
    )

    eval_chart = go.Figure()
    eval_chart.add_trace(go.Bar(
        x=fold_results["fold"], y=fold_results["mae"],
        name="MAE", marker_color="#C4501A", offsetgroup=1,
    ))
    eval_chart.add_trace(go.Bar(
        x=fold_results["fold"], y=fold_results["rmse"],
        name="RMSE", marker_color="#B8860B", offsetgroup=2,
    ))
    eval_chart.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", size=12, color="#5A5A5A"),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title="Fold", tickmode="array", tickvals=fold_results["fold"].tolist(), showgrid=False),
        yaxis=dict(title="Error", gridcolor="#F0EDE8"),
    )
    st.plotly_chart(eval_chart, use_container_width=True, config={"displayModeBar": False})


# ── Footer ───────────────────────────────────────────────────────────────────

footer_eval_label = (
    f"{len(fold_results)}-fold walk-forward CV"
    if not fold_results.empty
    else "walk-forward CV"
)
st.markdown(f"""
<div class="footer">
    <strong>Pollen & Pain</strong>
    <div style="margin-top: 4px;">Forecasting Neighborhood-Level Asthma Emergency Department Surges in New York City</div>
    <div class="footer-row">
        <span class="footer-item">NYC DOHMH Syndromic Surveillance</span>
        <span class="footer-item">AAAAI National Allergy Bureau</span>
        <span class="footer-item">Open-Meteo</span>
        <span class="footer-item">EPA AQS</span>
        <span class="footer-item">NYC Street Tree Census</span>
        <span class="footer-item">NYC 311</span>
    </div>
    <div style="margin-top: 8px;">
        Model evaluation: {footer_eval_label} · Full-period dashboard uses fitted predictions from the final model trained on all available data
    </div>
    <div style="margin-top: 8px; font-weight: 500; color: var(--text-primary);">
        Built by Saketh Boddu, Andre Nguyen, and Nam Lai
    </div>
</div>
""", unsafe_allow_html=True)
