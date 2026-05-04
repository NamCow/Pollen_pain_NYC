# Pollen & Pain

Forecasting neighborhood-level asthma emergency department surges in New York City.

Built by Saketh Boddu, Andre Nguyen, and Nam Lai.

---

## What this project does

Asthma is one of the most common reasons people in NYC end up in the emergency department, and certain neighborhoods get hit much harder than others. The problem is that most forecasting tools only work at the city or county level, which hides the variation between neighborhoods where public health decisions actually get made.

This project predicts monthly asthma ED visit counts for each of NYC's 197 neighborhoods (NTA level) across all five boroughs. It combines pollen data, weather, air quality, tree canopy density, and community health baselines to figure out which neighborhoods are most likely to see surges and when.

The end goal is a tool that schools, clinics, and public health planners can use to prepare before a surge happens rather than react after it does.

## How it works

The project has three main parts: a data pipeline, a machine learning model, and a dashboard.

### Data pipeline

We pull data from a bunch of different public sources, including the AAAAI pollen station, Open-Meteo weather archive, EPA air quality monitors, the NYC street tree census, NYC 311 complaints, and NYC DOHMH health surveys. Each source comes in a different geographic unit and time scale, so the ETL scripts align everything to the NTA neighborhood level on a monthly basis.

The target variable (monthly asthma ED visits per neighborhood) had to be constructed because NYC only publishes this data at the borough level. We distributed borough counts across neighborhoods proportionally using 2023 NTA-level ground-truth prevalence rates from DOHMH.

All processed data is stored in a PostgreSQL database hosted on Supabase. The dashboard reads directly from the database.

### Machine learning

We trained two models using walk-forward time-series cross-validation, which means the model only ever trains on past data and predicts forward, the same way it would work in the real world.

The primary model is an XGBoost regressor that predicts exact ED visit counts per neighborhood per month. The baseline is a logistic regression classifier that predicts whether a neighborhood will have an above-median or below-median month.

We also compared XGBoost against two naive baselines: a seasonal average (predict the historical mean for that calendar month) and a prior-month carry-forward (predict next month equals this month).

Key findings from the model:
- Neighborhood-level asthma prevalence is the strongest predictor of ED visit volume, which makes sense since it captures the baseline health burden.
- Pollen exposure at a 4-week lag is the next most important environmental driver, confirming the documented delay between pollen exposure and asthma ED surges.
- Temperature extremes, PM2.5, tree canopy density, and ozone all contribute meaningfully.

### Dashboard

The Streamlit dashboard connects live to the Supabase database and lets users explore predictions interactively. It includes a neighborhood risk map (choropleth), feature importance visualization, actual vs. predicted trend charts for individual neighborhoods, and a full model evaluation section with holdout metrics, baseline comparisons, and validation signals.

## Results

On a held-out test period (May to October 2025) that the model never saw during training or tuning:

- MAE of 1.98 ED visits (mean absolute error)
- RMSE of 2.36
- Pearson correlation of 0.973 between predicted and actual values
- 85.5% of neighborhoods were predicted within 15% error, exceeding the 70% target

The logistic regression baseline achieved a mean AUC-ROC of 0.885 across 4 folds.

The model outperformed the prior-month baseline on all metrics. It also outperformed the seasonal average on MAE and RMSE, though the seasonal average had a higher percentage of neighborhoods within 15% (91.4% vs 85.5%). This is because the seasonal average makes safe, flat predictions that are rarely far off, while XGBoost takes more risk trying to predict actual fluctuations and sometimes overshoots on individual neighborhoods.

We also validated the model against Community Health Survey prevalence rates. The correlation between predicted ED visits and CHS asthma prevalence was 0.894 (p < 0.001), nearly identical to the correlation between actual ED visits and CHS prevalence (0.888). This means the model learned real spatial health patterns, not noise.

Pollen lag correlations were statistically significant at both the 2-week lag (r = 0.098, p < 0.001) and 4-week lag (r = 0.155, p < 0.001), confirming the biological relationship between pollen exposure and asthma ED surges.

## How to run it

You need Python 3.11 or later and a `.env` file with the database credentials.

Set up the `.env` file in the project root or parent directory:

```
DB_HOST="..."
DB_PORT=5432
DB_NAME="postgres"
DB_USER="..."
DB_PASSWORD="..."
```

Install the dependencies:

```
pip install pandas psycopg2-binary streamlit folium streamlit-folium plotly geopandas python-dotenv xgboost scikit-learn
```

Run the dashboard:

```
streamlit run src/dashboard/app.py
```

It will open at http://localhost:8501.

## Data sources

- AAAAI National Allergy Bureau: daily pollen counts (tree, weed, grass, mold) from March 2022 onward
- Open-Meteo: daily temperature, precipitation, and wind speed for each neighborhood centroid
- EPA AQS and AirNow: monthly PM2.5, NO2, and ozone levels
- NYC DOHMH: asthma ED visit rates, community health survey respiratory data
- NYC Open Data: street tree census (683,788 trees), 311 air quality complaints
- NYC syndromic surveillance: monthly borough-level asthma ED counts

## Limitations and assumptions

There are several things worth being upfront about.

The biggest one is that the model's top feature by a wide margin is neighborhood-level asthma prevalence from the Community Health Survey, which accounts for about 67% of feature importance. This means the model is largely learning that high-asthma neighborhoods stay high, with environmental factors like pollen and weather driving the month-to-month variation on top of that baseline. The environmental forecasting value is real but incremental. If CHS prevalence data were unavailable, the model would perform significantly worse.

The target variable itself is an estimate. NYC does not publish monthly asthma ED visits at the neighborhood level, so we constructed it by distributing borough-level monthly counts across NTAs proportionally using annual prevalence rates. This assumes the within-borough distribution of ED visits is stable month to month, which may not hold during localized events like construction dust or wildfires.

Pollen data comes from a single monitoring station and is applied uniformly across all neighborhoods. In reality, pollen exposure varies with local vegetation, wind patterns, and building density. The tree canopy and weather features partially compensate for this, but true neighborhood-level pollen variation is not captured.

The modeling window covers March 2022 to October 2025, which is about 32 months of usable data across 186 NTAs. This is enough for the model to learn seasonal patterns but may not generalize to unusual years (e.g., pandemic-era healthcare utilization shifts, climate anomalies).

XGBoost hyperparameters were set manually rather than tuned with grid search or Bayesian optimization. The model performs well with the current settings, but there is likely room for marginal improvement with systematic tuning.

The model does not produce prediction intervals or uncertainty estimates. It outputs point predictions, which means a prediction of 30 ED visits for a neighborhood does not communicate whether the model is confident or uncertain about that number.

Finally, while the model outperforms the prior-month baseline on all metrics, it does not beat the seasonal average on the percentage of NTAs within 15% error. The seasonal average achieves 91.4% versus XGBoost's 85.5%. This is because the seasonal average makes conservative, flat predictions that are rarely far off, while XGBoost takes more risk by trying to predict actual monthly fluctuations. The tradeoff is lower total error at the cost of slightly less consistency across individual neighborhoods.

## Repository structure

- `config/` - centralized configuration (settings.yaml with all hyperparameters, paths, and feature definitions)
- `data/` - raw source data, processed CSVs, model outputs, and reference files (shapefiles, crosswalks)
- `src/extract/` - scripts that pull data from APIs, web scraping, and bulk downloads
- `src/etl/` - scripts that transform, aggregate, and align data to NTA level
- `src/features/` - feature engineering (pollen index, lag features, canopy score)
- `src/models/` - XGBoost and logistic regression training with walk-forward CV
- `src/database/` - schema setup and database loading scripts
- `src/dashboard/` - Streamlit application
- `src/utils/` - shared utilities including configuration loader

## References

1. Darrow, L. A., Hess, J., Rogers, C. A., Tolbert, P. E., Klein, M., & Sarnat, S. E. (2012). Ambient pollen concentrations and emergency department visits for asthma and wheeze. *Journal of Allergy and Clinical Immunology*, 130(3), 630–638. https://doi.org/10.1016/j.jaci.2012.06.020

2. Erbas, B., Jazayeri, M., Lambert, K. A., Katelaris, C. H., Prendergast, L. A., Tham, R., ... & Abramson, M. J. (2018). Outdoor pollen is a trigger of child and adolescent asthma emergency department presentations: A systematic review and meta-analysis. *Allergy*, 73(8), 1632–1641. https://doi.org/10.1111/all.13407

3. Gleason, J. A., Bielory, L., & Fagliano, J. A. (2014). Associations between ozone, PM2.5, and four pollen types on emergency department pediatric asthma events during the warm season in New Jersey: A case-crossover study. *Environmental Research*, 132, 421–429. https://doi.org/10.1016/j.envres.2014.03.035

4. NYC Department of Health and Mental Hygiene. (2023). *Community Health Survey: Neighborhood-level respiratory health indicators*. https://www.nyc.gov/site/doh/data/data-sets/community-health-survey.page

5. Zheng, X.-Y., Ding, H., Jiang, L.-N., Chen, S.-W., Zheng, J.-P., Qiu, M., ... & Guan, W.-J. (2015). Association between air pollutants and asthma emergency room visits and hospital admissions in time series studies: A systematic review and meta-analysis. *PLOS ONE*, 10(9), e0138146. https://doi.org/10.1371/journal.pone.0138146

6. Sheffield, P. E., Weinberger, K. R., Ito, K., Matte, T. D., Mathes, R. W., Robinson, G. S., & Kinney, P. L. (2011). The association of tree pollen concentration peaks and allergy medication sales in New York City: 2003–2008. *ISRN Allergy*, 2011, 537194. https://doi.org/10.5402/2011/537194

7. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785

8. Tashiro, H., & Shore, S. A. (2019). Obesity and severe asthma. *Allergology International*, 68(2), 135–142. https://doi.org/10.1016/j.alit.2018.10.004

9. Héguy, L., Garneau, M., Goldberg, M. S., Raphoz, M., Guay, F., & Valois, M.-F. (2008). Associations between grass and weed pollen and emergency department visits for asthma among children in Montreal. *Environmental Research*, 106(2), 203–211. https://doi.org/10.1016/j.envres.2007.10.005
