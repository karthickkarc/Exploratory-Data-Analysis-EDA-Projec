"""
Tests for eda_analysis.py
Run with: pytest tests/
"""

import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "../src")
from eda_analysis import (
    load_sample_data,
    statistical_summary,
    extract_insights,
)


@pytest.fixture
def sample_df():
    return load_sample_data()


def test_sample_data_shape(sample_df):
    assert sample_df.shape[0] == 500
    assert sample_df.shape[1] >= 6


def test_sample_data_columns(sample_df):
    expected = {"Age", "Experience", "Education", "Department",
                "Salary", "Performance", "Satisfaction"}
    assert expected.issubset(set(sample_df.columns))


def test_salary_range(sample_df):
    valid = sample_df["Salary"].dropna()
    assert valid.min() >= 20000
    assert valid.max() <= 160000


def test_statistical_summary_keys(sample_df):
    info = statistical_summary(sample_df)
    assert "numeric" in info
    assert "categorical" in info
    assert "missing" in info
    assert len(info["numeric"]) > 0


def test_insights_returns_list(sample_df):
    corr = sample_df[["Age", "Experience", "Salary"]].corr()
    insights = extract_insights(
        sample_df, corr,
        ["Age", "Experience", "Salary"],
        ["Education", "Department"]
    )
    assert isinstance(insights, list)
    assert len(insights) > 0


def test_insights_content(sample_df):
    corr = sample_df[["Age", "Experience", "Salary"]].corr()
    insights = extract_insights(
        sample_df, corr,
        ["Age", "Experience", "Salary"],
        ["Education", "Department"]
    )
    # At least one insight should mention a correlation
    assert any("correlation" in i.lower() for i in insights)


def test_no_duplicate_column_names(sample_df):
    assert len(sample_df.columns) == len(set(sample_df.columns))
