-- Pollen & Pain PostgreSQL Schema
-- Stores processed feature data and modeling tables.
-- Raw CSVs are loaded via src/database/load_to_postgres.py

DROP SCHEMA IF EXISTS pollen CASCADE;
CREATE SCHEMA pollen;

-- Reference: NTA neighborhoods
CREATE TABLE pollen.neighborhoods (
    nta_code       VARCHAR(10) PRIMARY KEY,
    nta_name       TEXT NOT NULL,
    borough        VARCHAR(20) NOT NULL,
    tree_count     INTEGER,
    total_dbh      NUMERIC,
    mean_dbh       NUMERIC,
    pct_good_health NUMERIC,
    chs_asthma_pct NUMERIC
);

-- Monthly pollen features (city-wide, one row per month)
CREATE TABLE pollen.pollen_monthly (
    year_month            VARCHAR(7) PRIMARY KEY,
    pollen_composite_avg  NUMERIC,
    pollen_composite_max  NUMERIC,
    tree_avg              NUMERIC,
    tree_max              NUMERIC,
    weed_avg              NUMERIC,
    grass_avg             NUMERIC,
    mold_avg              NUMERIC,
    pollen_14d_lag_avg    NUMERIC,
    pollen_28d_lag_avg    NUMERIC
);

-- Monthly weather per NTA
CREATE TABLE pollen.weather_monthly (
    nta_code       VARCHAR(10) NOT NULL REFERENCES pollen.neighborhoods(nta_code),
    year_month     VARCHAR(7) NOT NULL,
    temp_max_mean  NUMERIC,
    temp_min_mean  NUMERIC,
    precip_total   NUMERIC,
    wind_max_mean  NUMERIC,
    PRIMARY KEY (nta_code, year_month)
);

-- Monthly air quality (city-wide)
CREATE TABLE pollen.air_quality_monthly (
    year_month  VARCHAR(7) PRIMARY KEY,
    pm25_mean   NUMERIC,
    ozone_mean  NUMERIC,
    no2_mean    NUMERIC
);

-- Monthly asthma ED visits per NTA (target variable)
CREATE TABLE pollen.asthma_ed_monthly (
    nta_code       VARCHAR(10) NOT NULL REFERENCES pollen.neighborhoods(nta_code),
    year_month     VARCHAR(7) NOT NULL,
    borough        VARCHAR(20),
    borough_count  NUMERIC,
    n_ntas         INTEGER,
    estimated_count NUMERIC,
    PRIMARY KEY (nta_code, year_month)
);

-- Final modeling table (denormalized for fast queries)
CREATE TABLE pollen.modeling_table (
    nta_code              VARCHAR(10) NOT NULL,
    nta_name              TEXT,
    borough               VARCHAR(20),
    year_month            VARCHAR(7) NOT NULL,
    ed_visits             NUMERIC,
    temp_max_mean         NUMERIC,
    temp_min_mean         NUMERIC,
    precip_total          NUMERIC,
    wind_max_mean         NUMERIC,
    pollen_composite_avg  NUMERIC,
    pollen_composite_max  NUMERIC,
    tree_avg              NUMERIC,
    tree_max              NUMERIC,
    weed_avg              NUMERIC,
    grass_avg             NUMERIC,
    mold_avg              NUMERIC,
    pollen_14d_lag_avg    NUMERIC,
    pollen_28d_lag_avg    NUMERIC,
    pm25_mean             NUMERIC,
    ozone_mean            NUMERIC,
    no2_mean              NUMERIC,
    tree_count            INTEGER,
    total_dbh             NUMERIC,
    mean_dbh              NUMERIC,
    pct_good_health       NUMERIC,
    chs_asthma_pct        NUMERIC,
    PRIMARY KEY (nta_code, year_month)
);

-- 311 complaints monthly (validation signal)
CREATE TABLE pollen.complaints_monthly (
    borough         VARCHAR(20) NOT NULL,
    year_month      VARCHAR(7) NOT NULL,
    complaint_count INTEGER,
    PRIMARY KEY (borough, year_month)
);

-- Model predictions
CREATE TABLE pollen.model_predictions (
    nta_code    VARCHAR(10) NOT NULL,
    borough     VARCHAR(20),
    year_month  VARCHAR(7) NOT NULL,
    ed_visits   NUMERIC,
    pred        NUMERIC,
    fold        INTEGER,
    PRIMARY KEY (nta_code, year_month, fold)
);

-- Feature importance
CREATE TABLE pollen.feature_importance (
    feature     TEXT PRIMARY KEY,
    importance  NUMERIC
);

CREATE INDEX idx_weather_month ON pollen.weather_monthly(year_month);
CREATE INDEX idx_asthma_month ON pollen.asthma_ed_monthly(year_month);
CREATE INDEX idx_modeling_month ON pollen.modeling_table(year_month);
CREATE INDEX idx_modeling_borough ON pollen.modeling_table(borough);
