"""
Data Validation Module
Ensures dataset compatibility and handles missing columns
"""

import pandas as pd
import streamlit as st
from typing import Tuple, List, Dict

# =============================================================================
# REQUIRED COLUMNS DEFINITION
# =============================================================================

REQUIRED_COLUMNS = {
    'id': 'int64',
    'price': 'float64',
    'bedrooms': 'float64',
    'bathrooms': 'float64',
    'sqft_living': 'float64',
    'sqft_lot': 'float64',
    'condition': 'float64',
    'grade': 'float64',
    'yr_built': 'int64',
    'zipcode': 'int64',
    'lat': 'float64',
    'long': 'float64'
}

OPTIONAL_COLUMNS = {
    'date': 'object',
    'floors': 'float64',
    'waterfront': 'float64',
    'view': 'float64',
    'yr_renovated': 'float64',
    'sqft_above': 'float64',
    'sqft_basement': 'float64'
}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_dataset(df: pd.DataFrame) -> Tuple[bool, List[str], Dict]:
    """
    Validates uploaded dataset for required columns and basic quality.

    Args:
        df: Raw uploaded dataframe

    Returns:
        Tuple of (is_valid, error_messages, report_dict)
    """
    errors = []
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_required': [],
        'missing_optional': [],
        'empty_columns': [],
        'type_issues': []
    }

    # Check required columns
    for col in REQUIRED_COLUMNS.keys():
        if col not in df.columns:
            errors.append(f"❌ Missing required column: '{col}'")
            report['missing_required'].append(col)

    # Check optional columns
    for col in OPTIONAL_COLUMNS.keys():
        if col not in df.columns:
            report['missing_optional'].append(col)

    # Check for completely empty columns
    for col in df.columns:
        if df[col].isna().all():
            errors.append(f"⚠️ Column '{col}' is completely empty")
            report['empty_columns'].append(col)

    # Check critical columns have sufficient data
    critical_cols = ['price', 'sqft_living', 'zipcode', 'lat', 'long']
    for col in critical_cols:
        if col in df.columns:
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            if missing_pct > 50:
                errors.append(f"⚠️ Column '{col}' has {missing_pct:.1f}% missing values (too high)")

    is_valid = len(report['missing_required']) == 0

    return is_valid, errors, report


def prepare_dataset_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares validated dataset by ensuring correct types and handling missing optionals.

    Args:
        df: Validated dataframe

    Returns:
        Prepared dataframe
    """
    df_prep = df.copy()

    # Ensure correct types for required columns
    for col, dtype in REQUIRED_COLUMNS.items():
        if col in df_prep.columns:
            try:
                if dtype == 'int64':
                    df_prep[col] = df_prep[col].fillna(0).astype('int64')
                elif dtype == 'float64':
                    df_prep[col] = df_prep[col].astype('float64')
            except Exception as e:
                st.warning(f"Type conversion warning for '{col}': {str(e)}")

    # Add missing optional columns with defaults
    if 'date' not in df_prep.columns:
        df_prep['date'] = pd.NaT

    return df_prep


def display_validation_report(report: Dict):
    """
    Displays validation report in Streamlit UI.

    Args:
        report: Validation report dictionary
    """
    with st.expander("📋 Dataset Validation Report", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Rows", f"{report['total_rows']:,}")
            st.metric("Total Columns", report['total_columns'])

        with col2:
            st.metric("Missing Required", len(report['missing_required']))
            st.metric("Missing Optional", len(report['missing_optional']))

        if report['missing_optional']:
            st.info(f"ℹ️ Optional columns not found (will use defaults): {', '.join(report['missing_optional'])}")

        if report['empty_columns']:
            st.warning(f"⚠️ Empty columns detected: {', '.join(report['empty_columns'])}")