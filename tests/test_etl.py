"""
Tests for ETL / transform logic.

Covers the NTA allocation logic (allocate_ed_to_nta) and weather aggregation
patterns. All tests use synthetic data — no real files or shapefiles needed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.transform.allocate_ed_to_nta import geoid_to_nta_code


class TestGeoidToNtaCode:
    """Test the GeoID -> NTA code conversion function."""

    def test_bronx_geoid(self):
        assert geoid_to_nta_code(50101) == "BX0101"

    def test_brooklyn_geoid(self):
        assert geoid_to_nta_code(470101) == "BK0101"

    def test_manhattan_geoid(self):
        assert geoid_to_nta_code(610101) == "MN0101"

    def test_queens_geoid(self):
        assert geoid_to_nta_code(810101) == "QN0101"

    def test_staten_island_geoid(self):
        assert geoid_to_nta_code(850101) == "SI0101"

    def test_unknown_prefix_raises(self):
        with pytest.raises(ValueError, match="Unknown GeoID prefix"):
            geoid_to_nta_code(990101)


class TestNtaAllocationLogic:
    """
    Test that proportional allocation of borough counts to NTAs preserves totals.

    This replicates the core logic from allocate_ed_to_nta.py:
      estimated_count = borough_count * nta_share
    where nta_share = rate / boro_total_rate within a borough.
    """

    @staticmethod
    def _build_allocation_data():
        """Create synthetic NTA shares and monthly borough counts."""
        nta_shares = pd.DataFrame({
            "nta_code": ["BX0001", "BX0002", "BX0003", "BK0001", "BK0002"],
            "borough": ["Bronx", "Bronx", "Bronx", "Brooklyn", "Brooklyn"],
            "rate": [30.0, 50.0, 20.0, 40.0, 60.0],
        })
        boro_total = nta_shares.groupby("borough")["rate"].sum().rename("boro_total")
        nta_shares = nta_shares.merge(boro_total, on="borough")
        nta_shares["nta_share"] = nta_shares["rate"] / nta_shares["boro_total"]

        monthly_boro = pd.DataFrame({
            "borough": ["Bronx", "Bronx", "Brooklyn", "Brooklyn"],
            "year_month": ["2023-03", "2023-04", "2023-03", "2023-04"],
            "count": [1000.0, 800.0, 1500.0, 1200.0],
        })
        return nta_shares, monthly_boro

    def test_shares_sum_to_one_per_borough(self):
        """Within each borough, NTA shares must sum to 1.0."""
        nta_shares, _ = self._build_allocation_data()
        sums = nta_shares.groupby("borough")["nta_share"].sum()
        for boro, total in sums.items():
            assert abs(total - 1.0) < 1e-10, f"{boro} shares sum to {total}, not 1.0"

    def test_allocation_preserves_borough_totals(self):
        """Sum of NTA estimated counts must equal the original borough count."""
        nta_shares, monthly_boro = self._build_allocation_data()
        merged = nta_shares.merge(monthly_boro, on="borough")
        merged["estimated_count"] = merged["count"] * merged["nta_share"]

        nta_boro_sums = merged.groupby(["borough", "year_month"])["estimated_count"].sum()
        original = monthly_boro.set_index(["borough", "year_month"])["count"]

        for idx in original.index:
            np.testing.assert_allclose(
                nta_boro_sums.loc[idx], original.loc[idx],
                rtol=1e-10,
                err_msg=f"Allocation mismatch for {idx}",
            )

    def test_higher_rate_nta_gets_larger_share(self):
        """An NTA with a higher rate should receive more estimated visits."""
        nta_shares, monthly_boro = self._build_allocation_data()
        merged = nta_shares.merge(monthly_boro, on="borough")
        merged["estimated_count"] = merged["count"] * merged["nta_share"]

        bronx_mar = merged[
            (merged["borough"] == "Bronx") & (merged["year_month"] == "2023-03")
        ].set_index("nta_code")

        assert bronx_mar.loc["BX0002", "estimated_count"] > bronx_mar.loc["BX0001", "estimated_count"]
        assert bronx_mar.loc["BX0001", "estimated_count"] > bronx_mar.loc["BX0003", "estimated_count"]

    def test_allocation_produces_correct_row_count(self):
        """n_ntas * n_months rows per borough."""
        nta_shares, monthly_boro = self._build_allocation_data()
        merged = nta_shares.merge(monthly_boro, on="borough")
        # Bronx: 3 NTAs * 2 months = 6; Brooklyn: 2 NTAs * 2 months = 4 => total 10
        assert len(merged) == 10


class TestWeatherAggregation:
    """
    Test monthly weather aggregation logic (replicates the pattern in
    src/transform/weather.py without importing it, since the main() reads files).
    """

    @staticmethod
    def _build_daily_weather():
        """Create synthetic daily weather for 2 NTAs over 2 months."""
        dates = pd.date_range("2023-03-01", "2023-04-30", freq="D")
        rows = []
        for nta in ["BX0001", "BX0002"]:
            for d in dates:
                rows.append({
                    "nta_code": nta,
                    "time": d,
                    "temperature_2m_max": 20.0,
                    "temperature_2m_min": 10.0,
                    "precipitation_sum": 2.0,
                    "wind_speed_10m_max": 15.0,
                })
        return pd.DataFrame(rows)

    def test_monthly_aggregation_row_count(self):
        """2 NTAs * 2 months = 4 rows after aggregation."""
        df = self._build_daily_weather()
        df["year_month"] = df["time"].dt.to_period("M").astype(str)
        monthly = df.groupby(["nta_code", "year_month"], as_index=False).agg(
            temp_max_mean=("temperature_2m_max", "mean"),
            precip_total=("precipitation_sum", "sum"),
        )
        assert len(monthly) == 4

    def test_precip_total_is_sum_not_mean(self):
        """Precipitation should be summed across days, not averaged."""
        df = self._build_daily_weather()
        df["year_month"] = df["time"].dt.to_period("M").astype(str)
        monthly = df.groupby(["nta_code", "year_month"], as_index=False).agg(
            precip_total=("precipitation_sum", "sum"),
        )
        # March has 31 days, each with 2.0mm => 62.0
        march_row = monthly[
            (monthly["nta_code"] == "BX0001") & (monthly["year_month"] == "2023-03")
        ]
        assert march_row["precip_total"].iloc[0] == pytest.approx(62.0)

    def test_temp_mean_is_average_not_sum(self):
        """Temperature should be averaged across days."""
        df = self._build_daily_weather()
        df["year_month"] = df["time"].dt.to_period("M").astype(str)
        monthly = df.groupby(["nta_code", "year_month"], as_index=False).agg(
            temp_max_mean=("temperature_2m_max", "mean"),
        )
        # All days have 20.0, so the mean should be 20.0
        assert monthly["temp_max_mean"].iloc[0] == pytest.approx(20.0)
