-- Pollen & Pain: Neighborhood-Level Asthma ED Surge Forecasting
-- PostgreSQL schema — one row per NTA per month for time-series tables,
-- one row per NTA for static lookup tables.

CREATE TABLE IF NOT EXISTS nta_metadata (
    nta_code    VARCHAR(10)  PRIMARY KEY,
    nta_name    TEXT         NOT NULL,
    borough     TEXT         NOT NULL
);

CREATE TABLE IF NOT EXISTS asthma_ed_monthly (
    nta_code        VARCHAR(10)  NOT NULL REFERENCES nta_metadata(nta_code),
    year_month      CHAR(7)      NOT NULL,  -- 'YYYY-MM'
    borough_count   NUMERIC,
    n_ntas          INTEGER,
    estimated_count NUMERIC      NOT NULL,
    PRIMARY KEY (nta_code, year_month)
);

CREATE TABLE IF NOT EXISTS weather_monthly (
    nta_code        VARCHAR(10)  NOT NULL REFERENCES nta_metadata(nta_code),
    year_month      CHAR(7)      NOT NULL,
    temp_max_mean   NUMERIC,
    temp_min_mean   NUMERIC,
    precip_total    NUMERIC,
    wind_max_mean   NUMERIC,
    PRIMARY KEY (nta_code, year_month)
);

CREATE TABLE IF NOT EXISTS pollen_monthly (
    year_month              CHAR(7)  PRIMARY KEY,
    pollen_composite_avg    NUMERIC,
    pollen_composite_max    NUMERIC,
    tree_avg                NUMERIC,
    tree_max                NUMERIC,
    weed_avg                NUMERIC,
    grass_avg               NUMERIC,
    mold_avg                NUMERIC,
    pollen_14d_lag_avg      NUMERIC,
    pollen_28d_lag_avg      NUMERIC
);

CREATE TABLE IF NOT EXISTS air_quality_monthly (
    year_month  CHAR(7)  PRIMARY KEY,
    pm25_mean   NUMERIC,
    ozone_mean  NUMERIC,
    no2_mean    NUMERIC
);

CREATE TABLE IF NOT EXISTS tree_canopy (
    nta_code        VARCHAR(10)  PRIMARY KEY REFERENCES nta_metadata(nta_code),
    tree_count      INTEGER,
    total_dbh       NUMERIC,
    mean_dbh        NUMERIC,
    pct_good_health NUMERIC
);

CREATE TABLE IF NOT EXISTS chs_baseline (
    nta_code        VARCHAR(10)  PRIMARY KEY REFERENCES nta_metadata(nta_code),
    chs_asthma_pct  NUMERIC
);

CREATE TABLE IF NOT EXISTS model_predictions (
    nta_code    VARCHAR(10)  NOT NULL REFERENCES nta_metadata(nta_code),
    year_month  CHAR(7)      NOT NULL,
    ed_visits   NUMERIC,
    pred        NUMERIC      NOT NULL,
    fold        INTEGER,
    PRIMARY KEY (nta_code, year_month)
);

CREATE TABLE IF NOT EXISTS feature_importance (
    feature     TEXT    PRIMARY KEY,
    importance  NUMERIC NOT NULL
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_asthma_ed_month  ON asthma_ed_monthly (year_month);
CREATE INDEX IF NOT EXISTS idx_weather_month     ON weather_monthly    (year_month);
CREATE INDEX IF NOT EXISTS idx_predictions_month ON model_predictions  (year_month);
CREATE INDEX IF NOT EXISTS idx_asthma_ed_boro    ON asthma_ed_monthly (nta_code);
