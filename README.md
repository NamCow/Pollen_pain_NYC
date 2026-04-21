# Pollen & Pain: Forecasting Neighborhood-Level Asthma ED Surges in NYC

## Team
- Saketh Boddu
- Andre Nguyen
- Nam Lai

---

## 🎯 Project Overview
**Pollen & Pain** is a public health decision-support application built to forecast monthly asthma emergency department (ED) visit surges across New York City at the Neighborhood Tabulation Area (NTA) level. 

By analyzing the intersection of environmental triggers (tree, weed, grass, and mold pollen, as well as extreme weather events) and compounding factors like air quality (PM2.5, NO2, Ozone), baseline community health, and tree canopy density, this system provides actionable intelligence. It accurately identifies which neighborhoods are most at-risk of asthma ED surges, empowering NYC health officials to proactively deploy mobile inhaler clinics and issue targeted health interventions.

## 🧠 Technical Architecture

The application is broken down into three core components:

1. **The Data Pipeline (ETL & Database)**
   - **Sources:** Extracts data from the EPA, NYC Open Data (Tree Census, 311 Complaints), Open-Meteo, and AAAAI pollen counters.
   - **Transformation:** Interpolates and allocates borough-level asthma ED visit counts to the neighborhood level using accurate 2023 NTA-level ground-truth prevalence rates.
   - **Storage (Supabase):** The pipeline outputs 9 processed relational tables stored securely in a hosted **PostgreSQL database (Supabase)**.

2. **Machine Learning Pipeline**
   - **Walk-Forward Time-Series CV:** The models are strictly trained using past data blocks to predict future blocks, eliminating temporal data leakage.
   - **Models Used:** 
     * **XGBoost Regressor (Primary):** Learns complex, non-linear environmental interactions to accurately predict exact continuous ED visit counts per NTA.
     * **Logistic Regression (Baseline):** Predicts above-median vs. below-median risk levels to act as a comparative tool.
   - **Feature Discoveries:** We found that base neighborhood health dictates the standard volume of ED visits, but **4-week delayed pollen exposure** and minimum temperatures drive acute, month-to-month surges.

3. **Streamlit Dashboard**
   - A highly interactive, professional web interface built for policymakers.
   - Connects live to the Supabase database.
   - Features visual tools such as interactive Folium choropleth neighborhood risk maps, feature importance charts ("Why the model made this prediction"), and drill-down NTA tables.

---

## 🚀 How to Run the Application

### Prerequisites
- **Python 3.11+** is required.
- You will need a `.env` file securely configured to connect to the Supabase database.

### 1. Set Up Your Environment
Ensure your `.env` file is present in the main directory (or parent directory) and contains the following keys securely provided by an administrator:
```env
DB_HOST="..."
DB_PORT=5432
DB_NAME="postgres"
DB_USER="..."
DB_PASSWORD="..."
```

### 2. Install Dependencies
Make sure all necessary packages are installed. You can install them using pip:
```bash
pip install pandas psycopg2-binary streamlit folium streamlit-folium plotly geopandas python-dotenv xgboost scikit-learn
```

### 3. Run the Dashboard
To boot up the interactive dashboard and connect live to the database, run the following command from the root folder (`Pollen_pain_NYC`):

```bash
python -m streamlit run src/dashboard/app.py
```
*Note: If you run into "ModuleNotFoundError" issues on Mac, ensure you are invoking Streamlit via the exact python executable you installed the packages to (e.g., `/opt/homebrew/.../python3.13 -m streamlit run src/dashboard/app.py`).*

The app will instantly launch a local web server, and you can view the dashboard by opening `http://localhost:8501` in your browser.

---

## 📁 Repository Structure
- `data/`: Raw source data, processed CSVs, and generated model inferences.
- `src/extract/`: Scraping and API ingestion tools.
- `src/transform/`: Feature engineering, NTA allocation algorithms, and target variable building.
- `src/database/`: Schema configurations (`setup_schema.sql`) and database population scripts (`load_to_postgres.py`).
- `src/models/`: XGBoost and Logistic Regression training and walk-forward cross-validation logic.
- `src/dashboard/`: The `app.py` Streamlit application and CSS styling.
- `docs/`: Proposal files and project documentation.
