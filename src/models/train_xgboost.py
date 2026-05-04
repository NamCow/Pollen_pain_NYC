"""
Train the primary XGBoost regression model for monthly asthma ED visits per NTA.

Validation follows the project report:
1. Keep a final later-period holdout window.
2. Run walk-forward time-series CV on earlier data only.
3. Compare holdout performance against simple forecasting baselines.
4. Cross-check holdout predictions against CHS respiratory prevalence.

Input:  data/processed/modeling_table.csv
Output: data/models/xgboost_predictions.csv
        data/models/final_predictions.csv
        data/models/feature_importance.csv
        data/models/fold_results.csv
        data/models/xgboost_holdout_predictions.csv
        data/models/holdout_baseline_comparison.csv
        data/models/evaluation_summary.txt
"""

from pathlib import Path

import pandas as pd
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.features.feature_engineering import (
    FEATURE_COLS,
    MIN_AVG_MONTHLY_ED_VISITS,
    TARGET,
    XGBOOST_PARAMS,
    holdout_split,
    load_modeling_table,
    walk_forward_splits,
)
from src.utils.config import MODELS_DIR, HOLDOUT_MONTHS

OUTPUT_DIR = MODELS_DIR


def build_model() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(**XGBOOST_PARAMS)


def regression_metrics(actual: pd.Series, pred: pd.Series) -> dict[str, float]:
    corr, corr_p = pearsonr(actual, pred)
    return {
        "mae": mean_absolute_error(actual, pred),
        "rmse": root_mean_squared_error(actual, pred),
        "pearson_r": corr,
        "pearson_p": corr_p,
    }


def pct_ntas_within_threshold(
    df: pd.DataFrame,
    actual_col: str = TARGET,
    pred_col: str = "pred",
    threshold: float = 0.15,
) -> float:
    per_nta = df.groupby("nta_code").apply(
        lambda g: mean_absolute_error(g[actual_col], g[pred_col]) / g[actual_col].mean(),
        include_groups=False,
    )
    return (per_nta < threshold).mean()


def seasonal_average_baseline(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    month_keys = pd.to_datetime(train["year_month"] + "-01").dt.month
    seasonal = (
        train.assign(calendar_month=month_keys)
        .groupby(["nta_code", "calendar_month"])[TARGET]
        .mean()
        .rename("seasonal_avg_pred")
        .reset_index()
    )
    test_month = pd.to_datetime(test["year_month"] + "-01").dt.month
    merged = test[["nta_code"]].copy()
    merged["calendar_month"] = test_month
    merged = merged.merge(seasonal, on=["nta_code", "calendar_month"], how="left")
    fallback = train.groupby("nta_code")[TARGET].mean()
    return merged["seasonal_avg_pred"].fillna(test["nta_code"].map(fallback))


def prior_observation_baseline(full_df: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    prior_map = (
        full_df.sort_values(["nta_code", "year_month"])
        .assign(prior_ed=lambda d: d.groupby("nta_code")[TARGET].shift(1))
        .set_index(["nta_code", "year_month"])["prior_ed"]
    )
    overall_fallback = full_df.groupby("nta_code")[TARGET].mean()
    key = pd.MultiIndex.from_frame(test[["nta_code", "year_month"]])
    prior = prior_map.reindex(key)
    return prior.fillna(test["nta_code"].map(overall_fallback)).reset_index(drop=True)


def lag_signal_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ("pollen_14d_lag_avg", "pollen_28d_lag_avg"):
        valid = df[[TARGET, col]].dropna()
        corr, pval = pearsonr(valid[col], valid[TARGET])
        rows.append({"feature": col, "pearson_r": corr, "pearson_p": pval, "rows": len(valid)})
    return pd.DataFrame(rows)


def build_final_predictions(model: xgb.XGBRegressor, df: pd.DataFrame) -> pd.DataFrame:
    out = df[["nta_code", "NTAName", "borough", "year_month", TARGET]].copy()
    out["pred"] = model.predict(df[FEATURE_COLS]).clip(min=0)
    out["residual"] = out[TARGET] - out["pred"]
    out["fold"] = 0
    return out


def error_by_borough(holdout_out: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for borough, grp in holdout_out.groupby("borough"):
        m = regression_metrics(grp[TARGET], grp["pred"])
        m["pct_ntas_within_15pct"] = pct_ntas_within_threshold(grp)
        m["borough"] = borough
        m["n_ntas"] = grp["nta_code"].nunique()
        m["n_rows"] = len(grp)
        rows.append(m)
    return pd.DataFrame(rows)


def error_by_season(holdout_out: pd.DataFrame) -> pd.DataFrame:
    month_num = pd.to_datetime(holdout_out["year_month"] + "-01").dt.month
    season_map = {3: "Spring", 4: "Spring", 5: "Spring",
                  6: "Summer", 7: "Summer", 8: "Summer",
                  9: "Fall", 10: "Fall"}
    holdout_out = holdout_out.copy()
    holdout_out["season"] = month_num.map(season_map).fillna("Other")
    rows = []
    for season, grp in holdout_out.groupby("season"):
        if len(grp) < 2:
            continue
        m = regression_metrics(grp[TARGET], grp["pred"])
        m["pct_ntas_within_15pct"] = pct_ntas_within_threshold(grp)
        m["season"] = season
        m["months"] = sorted(grp["year_month"].unique())
        m["n_rows"] = len(grp)
        rows.append(m)
    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading modeling table...")
    df = load_modeling_table()
    df = df[["nta_code", "NTAName", "borough", "year_month", TARGET] + FEATURE_COLS].copy()
    print(f"  {len(df):,} rows | {df['nta_code'].nunique()} NTAs | {df['year_month'].nunique()} months")
    print(f"  Modeling window: {df['year_month'].min()} to {df['year_month'].max()}")
    print(f"  Minimum mean monthly ED filter: {MIN_AVG_MONTHLY_ED_VISITS:.0f}+ cases")

    dev_df, holdout_df, dev_end, holdout_start, holdout_end = holdout_split(df, holdout_months=HOLDOUT_MONTHS)
    print(f"  Development window: {dev_df['year_month'].min()} to {dev_end}")
    print(f"  Holdout window: {holdout_start} to {holdout_end}")

    print("\n=== Walk-Forward Cross-Validation (development only) ===")
    fold_results = []
    cv_preds = []

    for fold_i, (train, test, train_end, test_start, test_end) in enumerate(
        walk_forward_splits(dev_df, n_test_months=6), 1
    ):
        model = build_model()
        model.fit(train[FEATURE_COLS], train[TARGET])
        preds = model.predict(test[FEATURE_COLS]).clip(min=0)

        test_out = test[["nta_code", "NTAName", "borough", "year_month", TARGET]].copy()
        test_out["pred"] = preds
        test_out["fold"] = fold_i
        cv_preds.append(test_out)

        metrics = regression_metrics(test[TARGET], test_out["pred"])
        metrics["pct_ntas_within_15pct"] = pct_ntas_within_threshold(test_out)
        metrics.update(
            {
                "fold": fold_i,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "train_rows": len(train),
                "test_rows": len(test),
            }
        )
        fold_results.append(metrics)

        print(f"\nFold {fold_i}: train<={train_end} -> test {test_start}..{test_end}")
        print(f"  Train: {len(train):,} rows  Test: {len(test):,} rows")
        print(
            f"  XGBoost - MAE: {metrics['mae']:.2f}  RMSE: {metrics['rmse']:.2f}  "
            f"Pearson r: {metrics['pearson_r']:.3f} (p={metrics['pearson_p']:.2e})"
        )
        print(f"  NTAs within 15% MAE: {metrics['pct_ntas_within_15pct']:.1%}")

    fold_df = pd.DataFrame(fold_results)
    pred_df = pd.concat(cv_preds, ignore_index=True)
    pred_df.to_csv(OUTPUT_DIR / "xgboost_predictions.csv", index=False)
    fold_df.to_csv(OUTPUT_DIR / "fold_results.csv", index=False)

    print("\n=== Final Holdout Evaluation ===")
    holdout_model = build_model()
    holdout_model.fit(dev_df[FEATURE_COLS], dev_df[TARGET])

    holdout_out = holdout_df[["nta_code", "NTAName", "borough", "year_month", TARGET, "chs_asthma_pct"]].copy()
    holdout_out = holdout_out.reset_index(drop=True)
    holdout_out["pred"] = holdout_model.predict(holdout_df[FEATURE_COLS]).clip(min=0)
    holdout_metrics = regression_metrics(holdout_out[TARGET], holdout_out["pred"])
    holdout_metrics["pct_ntas_within_15pct"] = pct_ntas_within_threshold(holdout_out)

    seasonal_pred = seasonal_average_baseline(dev_df, holdout_df)
    prior_pred = prior_observation_baseline(df, holdout_df)
    baseline_rows = []
    for baseline_name, baseline_pred in (
        ("seasonal_average", seasonal_pred),
        ("prior_observation", prior_pred),
    ):
        baseline_metrics = regression_metrics(holdout_df[TARGET], baseline_pred)
        baseline_out = holdout_out[["nta_code", "year_month", TARGET]].copy()
        baseline_out["pred"] = pd.Series(baseline_pred).to_numpy()
        baseline_metrics["pct_ntas_within_15pct"] = pct_ntas_within_threshold(baseline_out)
        baseline_metrics["baseline"] = baseline_name
        baseline_rows.append(baseline_metrics)
    baseline_df = pd.DataFrame(baseline_rows)

    chs_eval = (
        holdout_out.groupby("nta_code", as_index=False)
        .agg(
            pred_mean=("pred", "mean"),
            actual_mean=(TARGET, "mean"),
            chs_asthma_pct=("chs_asthma_pct", "first"),
        )
    )
    pred_chs_r, pred_chs_p = pearsonr(chs_eval["pred_mean"], chs_eval["chs_asthma_pct"])
    actual_chs_r, actual_chs_p = pearsonr(chs_eval["actual_mean"], chs_eval["chs_asthma_pct"])

    holdout_out.to_csv(OUTPUT_DIR / "xgboost_holdout_predictions.csv", index=False)
    baseline_df.to_csv(OUTPUT_DIR / "holdout_baseline_comparison.csv", index=False)

    print(
        f"  Holdout XGBoost - MAE: {holdout_metrics['mae']:.2f}  RMSE: {holdout_metrics['rmse']:.2f}  "
        f"Pearson r: {holdout_metrics['pearson_r']:.3f} (p={holdout_metrics['pearson_p']:.2e})"
    )
    print(f"  Holdout NTAs within 15% MAE: {holdout_metrics['pct_ntas_within_15pct']:.1%}")
    for _, row in baseline_df.iterrows():
        print(
            f"  Baseline {row['baseline']}: MAE {row['mae']:.2f}  RMSE {row['rmse']:.2f}  "
            f"Pearson r {row['pearson_r']:.3f}"
        )
    print(f"  CHS cross-check (pred vs CHS): r={pred_chs_r:.3f} (p={pred_chs_p:.2e})")
    print(f"  CHS cross-check (actual vs CHS): r={actual_chs_r:.3f} (p={actual_chs_p:.2e})")

    print("\n=== Error Analysis by Subgroup ===")
    borough_error_df = error_by_borough(holdout_out)
    season_error_df = error_by_season(holdout_out)
    borough_error_df.to_csv(OUTPUT_DIR / "holdout_error_by_borough.csv", index=False)
    season_error_df.to_csv(OUTPUT_DIR / "holdout_error_by_season.csv", index=False)

    print("\n  By Borough:")
    for _, row in borough_error_df.iterrows():
        print(
            f"    {row['borough']:15s} MAE={row['mae']:.2f}  RMSE={row['rmse']:.2f}  "
            f"r={row['pearson_r']:.3f}  within 15%={row['pct_ntas_within_15pct']:.1%}  "
            f"({int(row['n_ntas'])} NTAs)"
        )
    print("\n  By Season:")
    for _, row in season_error_df.iterrows():
        print(
            f"    {row['season']:10s} MAE={row['mae']:.2f}  RMSE={row['rmse']:.2f}  "
            f"r={row['pearson_r']:.3f}  within 15%={row['pct_ntas_within_15pct']:.1%}"
        )

    print("\n=== Training Production Model (all filtered data) ===")
    final_model = build_model()
    final_model.fit(df[FEATURE_COLS], df[TARGET])

    importance = pd.DataFrame(
        {"feature": FEATURE_COLS, "importance": final_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    final_predictions = build_final_predictions(final_model, df)
    final_predictions.to_csv(OUTPUT_DIR / "final_predictions.csv", index=False)

    lag_corr_df = lag_signal_correlations(df)

    summary_lines = [
        "=== EVALUATION SUMMARY ===",
        f"Filtered rows: {len(df):,}  NTAs: {df['nta_code'].nunique()}  Months: {df['year_month'].nunique()}",
        f"Modeling window: {df['year_month'].min()} to {df['year_month'].max()}",
        f"Development window: {dev_df['year_month'].min()} to {dev_end}",
        f"Holdout window: {holdout_start} to {holdout_end}",
        f"NTA inclusion rule: mean monthly ED visits >= {MIN_AVG_MONTHLY_ED_VISITS:.0f}",
        "",
        "--- XGBoost Regression (walk-forward CV on development window) ---",
        f"Folds: {len(fold_df)}",
        f"Mean MAE:  {fold_df['mae'].mean():.2f}",
        f"Mean RMSE: {fold_df['rmse'].mean():.2f}",
        f"Mean Pearson r: {fold_df['pearson_r'].mean():.3f}",
        f"Mean NTAs within 15%: {fold_df['pct_ntas_within_15pct'].mean():.1%}",
        "",
        "--- Final Holdout Evaluation ---",
        f"XGBoost MAE:  {holdout_metrics['mae']:.2f}",
        f"XGBoost RMSE: {holdout_metrics['rmse']:.2f}",
        f"XGBoost Pearson r: {holdout_metrics['pearson_r']:.3f} (p={holdout_metrics['pearson_p']:.2e})",
        f"XGBoost NTAs within 15%: {holdout_metrics['pct_ntas_within_15pct']:.1%}",
        "",
        "--- Holdout Baseline Comparison ---",
    ]
    for _, row in baseline_df.iterrows():
        summary_lines.append(
            f"  {row['baseline']:17s} MAE={row['mae']:.2f}  RMSE={row['rmse']:.2f}  "
            f"Pearson r={row['pearson_r']:.3f}  NTAs within 15%={row['pct_ntas_within_15pct']:.1%}"
        )

    summary_lines.extend(
        [
            "",
            "--- CHS Cross-Check ---",
            f"Predicted holdout mean vs CHS prevalence: r={pred_chs_r:.3f} (p={pred_chs_p:.2e})",
            f"Actual holdout mean vs CHS prevalence:    r={actual_chs_r:.3f} (p={actual_chs_p:.2e})",
            "",
            "--- Pollen Lag Correlations ---",
        ]
    )
    for _, row in lag_corr_df.iterrows():
        summary_lines.append(
            f"  {row['feature']:20s} r={row['pearson_r']:.3f} (p={row['pearson_p']:.2e}, n={int(row['rows'])})"
        )

    summary_lines.extend(["", "--- Error by Borough (holdout) ---"])
    for _, row in borough_error_df.iterrows():
        summary_lines.append(
            f"  {row['borough']:15s} MAE={row['mae']:.2f}  RMSE={row['rmse']:.2f}  "
            f"r={row['pearson_r']:.3f}  within 15%={row['pct_ntas_within_15pct']:.1%}  "
            f"({int(row['n_ntas'])} NTAs)"
        )

    summary_lines.extend(["", "--- Error by Season (holdout) ---"])
    for _, row in season_error_df.iterrows():
        summary_lines.append(
            f"  {row['season']:10s} MAE={row['mae']:.2f}  RMSE={row['rmse']:.2f}  "
            f"r={row['pearson_r']:.3f}  within 15%={row['pct_ntas_within_15pct']:.1%}"
        )

    summary_lines.extend(["", "--- Feature Importance (top 10) ---"])
    for _, row in importance.head(10).iterrows():
        summary_lines.append(f"  {row['feature']:30s} {row['importance']:.4f}")

    summary = "\n".join(summary_lines)
    (OUTPUT_DIR / "evaluation_summary.txt").write_text(summary)

    print(f"\n{summary}")
    print(f"\nSaved outputs to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
