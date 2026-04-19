"""
Pollen & Pain — Streamlit Dashboard
Forecasting neighborhood-level asthma ED surges in NYC.
"""

import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[2]
PREDS_CV     = ROOT / "data/models/xgboost_predictions.csv"
PREDS_FINAL  = ROOT / "data/models/final_predictions.csv"
IMPORTANCE   = ROOT / "data/models/feature_importance.csv"
FOLD_RESULTS = ROOT / "data/models/fold_results.csv"
MODELING     = ROOT / "data/processed/modeling_table.csv"
NTA_SHP      = ROOT / "data/Input/nynta2020.shp"


# ── data loaders ────────────────────────────────────────────────────────────
@st.cache_data
def load_predictions():
    path = PREDS_FINAL if PREDS_FINAL.exists() else PREDS_CV
    df = pd.read_csv(path)
    if "pred" not in df.columns:
        df["pred"] = df.get("pred", df.get("estimated_count", 0))
    if "ed_visits" not in df.columns:
        df["ed_visits"] = df.get("ed_visits", None)
    return df


@st.cache_data
def load_importance():
    return pd.read_csv(IMPORTANCE)


@st.cache_data
def load_fold_results():
    return pd.read_csv(FOLD_RESULTS)


@st.cache_data
def load_geojson():
    try:
        import geopandas as gpd
        gdf = gpd.read_file(NTA_SHP).to_crs(epsg=4326)
        gdf = gdf[gdf["NTAType"] == "0"][["NTA2020", "NTAName", "BoroName", "geometry"]].copy()
        return json.loads(gdf.to_json()), gdf[["NTA2020", "NTAName", "BoroName"]].rename(
            columns={"NTA2020": "nta_code", "BoroName": "borough"}
        )
    except Exception:
        return None, None


# ── layout ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pollen & Pain NYC",
    page_icon="🌿",
    layout="wide",
)

st.title("Pollen & Pain: Asthma ED Surge Forecasting — NYC")
st.caption("Monthly neighborhood-level predictions using pollen, weather, air quality, tree canopy & CHS data.")

preds   = load_predictions()
imp_df  = load_importance()
fold_df = load_fold_results()

months  = sorted(preds["year_month"].unique())
boroughs = ["All"] + sorted(preds["borough"].unique())

# ── sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    selected_borough = st.selectbox("Borough", boroughs)
    selected_month   = st.selectbox("Month", months, index=len(months) - 1)

    st.markdown("---")
    st.markdown("**Model CV Performance**")
    st.metric("Mean MAE",      f"{fold_df['mae'].mean():.2f} visits/NTA")
    st.metric("Mean Pearson r", f"{fold_df['pearson_r'].mean():.3f}")
    if "lr_auc" in fold_df.columns:
        st.metric("LR AUC-ROC", f"{fold_df['lr_auc'].mean():.3f}")

# ── filter data ─────────────────────────────────────────────────────────────
month_df = preds[preds["year_month"] == selected_month].copy()
if selected_borough != "All":
    month_df = month_df[month_df["borough"] == selected_borough]

# ── top summary metrics ──────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("NTAs shown",       f"{month_df['nta_code'].nunique()}")
col2.metric("Avg predicted",    f"{month_df['pred'].mean():.1f}")
col3.metric("Highest predicted", f"{month_df['pred'].max():.1f}")
if "ed_visits" in month_df.columns and month_df["ed_visits"].notna().any():
    mae = (month_df["pred"] - month_df["ed_visits"]).abs().mean()
    col4.metric("Month MAE", f"{mae:.2f}")

st.markdown("---")

# ── map + feature importance ─────────────────────────────────────────────────
map_col, imp_col = st.columns([3, 1])

with map_col:
    st.subheader(f"Predicted Asthma ED Visits — {selected_month}")
    geojson, nta_meta = load_geojson()

    if geojson is not None:
        fig_map = px.choropleth_mapbox(
            month_df,
            geojson=geojson,
            locations="nta_code",
            featureidkey="properties.NTA2020",
            color="pred",
            color_continuous_scale="YlOrRd",
            mapbox_style="carto-positron",
            zoom=9.5,
            center={"lat": 40.7128, "lon": -74.006},
            opacity=0.7,
            hover_data={"nta_code": True, "pred": ":.1f", "borough": True},
            labels={"pred": "Predicted ED visits", "nta_code": "NTA"},
        )
        fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=480)
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("Shapefile not found — showing bar chart instead.")
        fig_bar = px.bar(
            month_df.sort_values("pred", ascending=False).head(30),
            x="nta_code", y="pred",
            color="borough",
            labels={"pred": "Predicted ED visits", "nta_code": "NTA"},
        )
        st.plotly_chart(fig_bar, use_container_width=True)

with imp_col:
    st.subheader("Feature Importance")
    top_imp = imp_df.head(10).sort_values("importance")
    fig_imp = px.bar(
        top_imp,
        x="importance",
        y="feature",
        orientation="h",
        labels={"importance": "Importance", "feature": ""},
        color="importance",
        color_continuous_scale="Blues",
    )
    fig_imp.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        margin={"t": 10, "b": 10},
        height=400,
    )
    st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("---")

# ── time series for selected NTA ─────────────────────────────────────────────
st.subheader("Time Series: Actual vs Predicted")

nta_options = sorted(preds["nta_code"].unique())
if selected_borough != "All":
    nta_options = sorted(preds[preds["borough"] == selected_borough]["nta_code"].unique())

selected_nta = st.selectbox("Select NTA", nta_options)

nta_ts = preds[preds["nta_code"] == selected_nta].sort_values("year_month")
nta_name = nta_ts["NTAName"].iloc[0] if "NTAName" in nta_ts.columns else selected_nta

fig_ts = go.Figure()
if "ed_visits" in nta_ts.columns and nta_ts["ed_visits"].notna().any():
    fig_ts.add_trace(go.Scatter(
        x=nta_ts["year_month"], y=nta_ts["ed_visits"],
        name="Actual", mode="lines+markers",
        line={"color": "#1f77b4", "width": 2},
    ))
fig_ts.add_trace(go.Scatter(
    x=nta_ts["year_month"], y=nta_ts["pred"],
    name="Predicted", mode="lines+markers",
    line={"color": "#ff7f0e", "width": 2, "dash": "dot"},
))
fig_ts.update_layout(
    title=f"{nta_name} ({selected_nta})",
    xaxis_title="Month",
    yaxis_title="Asthma ED Visits (estimated)",
    hovermode="x unified",
    height=380,
)
st.plotly_chart(fig_ts, use_container_width=True)

st.markdown("---")

# ── CV fold performance table ────────────────────────────────────────────────
with st.expander("Walk-Forward CV Fold Results"):
    display_cols = ["fold", "test_start", "test_end", "train_rows", "test_rows",
                    "mae", "rmse", "pearson_r"]
    if "lr_auc" in fold_df.columns:
        display_cols += ["lr_auc", "lr_accuracy"]
    st.dataframe(
        fold_df[[c for c in display_cols if c in fold_df.columns]]
        .style.format({
            "mae": "{:.2f}", "rmse": "{:.2f}",
            "pearson_r": "{:.3f}", "lr_auc": "{:.3f}",
            "lr_accuracy": "{:.2%}",
        }),
        use_container_width=True,
    )

# ── raw predictions table ────────────────────────────────────────────────────
with st.expander(f"Raw predictions — {selected_month}"):
    show_cols = ["nta_code", "NTAName", "borough", "year_month", "pred"]
    if "ed_visits" in month_df.columns:
        show_cols.append("ed_visits")
    st.dataframe(
        month_df[show_cols].sort_values("pred", ascending=False),
        use_container_width=True,
    )
