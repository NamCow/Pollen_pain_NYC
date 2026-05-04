"""
Load processed CSVs into PostgreSQL pollen schema.

Requires: psycopg2-binary, pandas
Usage:    python src/database/load_to_postgres.py

Assumes the schema from setup_schema.sql has already been created.
Connection defaults to local pollen_pain database.
"""

import os

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from pathlib import Path
from dotenv import load_dotenv

PROCESSED = Path("./data/processed")

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

DB_PARAMS = dict(
    dbname=os.environ.get("DB_NAME", "postgres"),
    user=os.environ.get("DB_USER", "postgres"),
    password=os.environ.get("DB_PASSWORD", ""),
    host=os.environ.get("DB_HOST", "localhost"),
    port=int(os.environ.get("DB_PORT", 5432)),
    sslmode="require" if os.environ.get("DB_HOST") not in (None, "", "localhost", "127.0.0.1") else "prefer",
)


def load_df(cur, table: str, df: pd.DataFrame):
    if df.empty:
        print(f"  SKIP {table} — empty dataframe")
        return
    cols = list(df.columns)
    col_str = ", ".join(cols)
    template = "(" + ", ".join(["%s"] * len(cols)) + ")"
    values = [tuple(None if pd.isna(v) else v for v in row) for row in df.itertuples(index=False, name=None)]
    cur.execute(f"TRUNCATE {table} CASCADE")
    execute_values(cur, f"INSERT INTO {table} ({col_str}) VALUES %s", values, template=template, page_size=1000)
    print(f"  {table}: {len(values):,} rows loaded")


def main():
    conn = psycopg2.connect(**DB_PARAMS)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        # 1. Neighborhoods (from tree canopy + CHS baseline)
        canopy = pd.read_csv(PROCESSED / "tree_canopy_by_nta.csv")
        chs = pd.read_csv(PROCESSED / "chs_baseline_by_nta.csv")
        neighborhoods = canopy.merge(chs[["nta_code", "NTAName", "borough", "chs_asthma_pct"]], on="nta_code", how="outer")
        neighborhoods = neighborhoods.rename(columns={"NTAName": "nta_name"})
        neighborhoods = neighborhoods[["nta_code", "nta_name", "borough", "tree_count", "total_dbh", "mean_dbh", "pct_good_health", "chs_asthma_pct"]]
        load_df(cur, "pollen.neighborhoods", neighborhoods)

        # 2. Pollen monthly
        pollen = pd.read_csv(PROCESSED / "pollen_monthly_features.csv")
        load_df(cur, "pollen.pollen_monthly", pollen)

        # 3. Weather monthly
        weather = pd.read_csv(PROCESSED / "weather_monthly_by_nta.csv")
        valid_ntas = set(neighborhoods["nta_code"])
        weather = weather[weather["nta_code"].isin(valid_ntas)]
        load_df(cur, "pollen.weather_monthly", weather)

        # 4. Air quality monthly
        aq = pd.read_csv(PROCESSED / "air_quality_monthly.csv")
        aq = aq[["year_month", "pm25_mean", "ozone_mean", "no2_mean"]]
        load_df(cur, "pollen.air_quality_monthly", aq)

        # 5. Asthma ED monthly
        ed = pd.read_csv(PROCESSED / "asthma_ed_monthly_nta.csv")
        ed = ed.rename(columns={"NTA2020": "nta_code"})
        ed = ed[["nta_code", "year_month", "borough", "borough_count", "n_ntas", "estimated_count"]]
        ed = ed[ed["nta_code"].isin(valid_ntas)]
        load_df(cur, "pollen.asthma_ed_monthly", ed)

        # 6. Modeling table
        modeling = pd.read_csv(PROCESSED / "modeling_table.csv")
        modeling = modeling.rename(columns={"NTAName": "nta_name"})
        modeling = modeling[["nta_code", "nta_name", "borough", "year_month", "ed_visits",
                             "temp_max_mean", "temp_min_mean", "precip_total", "wind_max_mean",
                             "pollen_composite_avg", "pollen_composite_max",
                             "tree_avg", "tree_max", "weed_avg", "grass_avg", "mold_avg",
                             "pollen_14d_lag_avg", "pollen_28d_lag_avg",
                             "pm25_mean", "ozone_mean", "no2_mean",
                             "tree_count", "total_dbh", "mean_dbh", "pct_good_health", "chs_asthma_pct"]]
        load_df(cur, "pollen.modeling_table", modeling)

        # 7. 311 complaints
        complaints = pd.read_csv(PROCESSED / "311_monthly_by_borough.csv")
        load_df(cur, "pollen.complaints_monthly", complaints)

        # 8. Model predictions & feature importance (load if they exist)
        models_dir = Path("./data/models")
        final_preds_path = models_dir / "final_predictions.csv"
        cv_preds_path = models_dir / "xgboost_predictions.csv"
        if final_preds_path.exists():
            preds = pd.read_csv(final_preds_path)
            preds = preds.rename(columns={"NTAName": "nta_name"})
            preds = preds[["nta_code", "borough", "year_month", "ed_visits", "pred", "fold"]]
            load_df(cur, "pollen.model_predictions", preds)
        elif cv_preds_path.exists():
            preds = pd.read_csv(cv_preds_path)
            preds = preds[["nta_code", "borough", "year_month", "ed_visits", "pred", "fold"]]
            load_df(cur, "pollen.model_predictions", preds)

        fi_path = models_dir / "feature_importance.csv"
        if fi_path.exists():
            fi = pd.read_csv(fi_path)
            load_df(cur, "pollen.feature_importance", fi)

        conn.commit()
        print("\nAll tables loaded successfully.")

    except Exception as e:
        conn.rollback()
        print(f"\nERROR — rolled back: {e}")
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
