"""
Data Preparation Module
Chunks 0-2 from Colab notebook
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple


# =============================================================================
# CHUNK 1: DATA CLEANING & IMPUTATION
# =============================================================================
@st.cache_data(show_spinner="🧹 Cleaning data...")
def chunk_01_data_cleaning(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Performs data cleaning and smart imputation (Chunk 1 from Colab).

    Strategy:
    - Replace 0 values with NaN for specific columns
    - Impute using zipcode-level medians (location-aware)
    - Fallback to city median if zipcode has no data

    Args:
        df: Raw dataframe

    Returns:
        tuple: (cleaned_df, audit_report_dict)
    """
    df_clean = df.copy()
    audit_report = {}

    # Convert date to datetime
    df_clean['date'] = pd.to_datetime(df_clean['date'])

    # Create renovation flag
    df_clean['is_renovated'] = df_clean['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)

    # Identify missing value columns
    cols_to_fix_zero = ['bedrooms', 'bathrooms', 'sqft_lot', 'yr_built']
    cols_to_fix_nan = ['condition', 'grade',]

    # Replace 0 with NaN where 0 means "missing"
    for col in cols_to_fix_zero:
        zeros_count = len(df_clean[df_clean[col] == 0])
        if zeros_count > 0:
            df_clean[col] = df_clean[col].replace(0, np.nan)
            audit_report[f'{col}_zeros_found'] = zeros_count

        # --- LOGIC PATCH START: sqft_above calculation ---
        # Logic: sqft_living = sqft_above + sqft_basement
        # We calculate missing 'sqft_above' mathematically instead of guessing with median.
        if 'sqft_above' in df_clean.columns and 'sqft_basement' in df_clean.columns:
            # 1. Fill NaNs using the formula
            df_clean['sqft_above'] = df_clean['sqft_above'].fillna(
                df_clean['sqft_living'] - df_clean['sqft_basement']
            )

            # 2. Safety check: ensure no negative values (clamp to 0)
            df_clean['sqft_above'] = df_clean['sqft_above'].apply(lambda x: max(x, 0) if pd.notnull(x) else x)
        # --- LOGIC PATCH END ---

    # Smart Imputation: Zipcode-aware
    all_cols_to_impute = cols_to_fix_zero + cols_to_fix_nan

    for col in all_cols_to_impute:
        before_nan = df_clean[col].isna().sum()
        if before_nan > 0:
            # Primary: Zipcode median
            df_clean[col] = df_clean[col].fillna(
                df_clean.groupby('zipcode')[col].transform('median')
            )
            # Fallback: City median
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

            after_nan = df_clean[col].isna().sum()
            audit_report[f'{col}_imputed'] = before_nan - after_nan

    # Duplicate detection
    n_duplicates = df_clean.duplicated().sum()
    if n_duplicates > 0:
        df_clean = df_clean.drop_duplicates()
        audit_report['duplicates_removed'] = n_duplicates

    # Duplicate IDs check
    n_dup_ids = df_clean['id'].duplicated().sum()
    audit_report['duplicate_ids'] = n_dup_ids

    audit_report['final_row_count'] = len(df_clean)
    audit_report['original_row_count'] = len(df)

    return df_clean, audit_report


# =============================================================================
# CHUNK 2: MARKET CONTEXT (METRICS ENGINE)
# =============================================================================
@st.cache_data(show_spinner="📊 Computing market metrics...")
def chunk_02_market_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates market context metrics (Chunk 2 from Colab).

    Metrics:
    - Price Gap: Difference from zipcode average
    - Price per Sqft
    - Sqft Gap: Efficiency metric
    - Density Type: Property classification by lot ratio

    Args:
        df: Cleaned dataframe

    Returns:
        DataFrame with added metric columns
    """
    df_context = df.copy()

    # A. Price Gap (Absolute $)
    zip_stats = df_context.groupby('zipcode')['price'].median().reset_index()
    zip_stats.rename(columns={'price': 'zip_avg_price'}, inplace=True)
    df_context = df_context.merge(zip_stats, on='zipcode', how='left')
    df_context['price_gap'] = df_context['zip_avg_price'] - df_context['price']

    # B. Price per Sqft & Efficiency Gap
    df_context['price_per_sqft'] = df_context['price'] / df_context['sqft_living']

    # Neighborhood Avg $/sqft
    zip_sqft_stats = df_context.groupby('zipcode')['price_per_sqft'].mean().reset_index()
    zip_sqft_stats.rename(columns={'price_per_sqft': 'zip_avg_price_sqft'}, inplace=True)
    df_context = df_context.merge(zip_sqft_stats, on='zipcode', how='left')

    df_context['sqft_gap'] = df_context['zip_avg_price_sqft'] - df_context['price_per_sqft']

    # C. Density Type (Lot vs Living Ratio)
    df_context['lot_living_ratio'] = df_context['sqft_lot'] / df_context['sqft_living']

    def categorize_density(ratio):
        if ratio < 1.0:
            return 'Condo/Townhouse (High Density)'
        elif ratio < 4:
            return 'Small Yard (Suburban)'
        else:
            return 'Large Lot (Villa/Estate)'

    df_context['density_type'] = df_context['lot_living_ratio'].apply(categorize_density)

    return df_context