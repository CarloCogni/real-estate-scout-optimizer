"""
Statistical Analysis Module
Chunks 3.x from Colab notebook

This module contains all exploratory and statistical analysis functions.
Each function corresponds to a specific chunk in the original Colab notebook.
"""

import pandas as pd
import numpy as np
import streamlit as st
from scipy.stats import trim_mean, chi2_contingency, f_oneway
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker


# =============================================================================
# CHUNK 3.1: UNIVARIATE ANALYSIS (MARKET DISTRIBUTION)
# =============================================================================

@st.cache_data(show_spinner="📊 Analyzing market distribution...")
def chunk_03_01_univariate_analysis(df: pd.DataFrame) -> Dict:
    """
    Analyzes market price distribution with skewness and kurtosis (Chunk 3.1 from Colab).

    CACHED: Statistical calculations are deterministic.

    Args:
        df: Market context dataframe with 'price' column

    Returns:
        Dictionary with statistics and interpretation
    """
    results = {}

    # Calculate metrics
    results['skewness'] = df['price'].skew()
    results['kurtosis'] = df['price'].kurtosis()
    results['mean'] = df['price'].mean()
    results['trimmed_mean'] = trim_mean(df['price'], proportiontocut=0.1)
    results['median'] = df['price'].median()
    results['std'] = df['price'].std()
    results['cv'] = (results['std'] / results['mean']) * 100  # Coefficient of variation

    # Opportunity distribution
    results['price_gap_mean'] = df['price_gap'].mean()
    results['price_gap_median'] = df['price_gap'].median()
    results['price_gap_positive_pct'] = (df['price_gap'] > 0).sum() / len(df) * 100

    # Interpretation flags
    results['is_skewed'] = abs(results['skewness']) > 1
    results['is_leptokurtic'] = results['kurtosis'] > 3
    results['high_volatility'] = results['cv'] > 50

    return results


# =============================================================================
# CHUNK 3.2: DEEP DIVE - SIZE TRAP ANALYSIS
# =============================================================================

@st.cache_data(show_spinner="🔍 Analyzing size vs value relationship...")
def chunk_03_02_size_trap_analysis(df: pd.DataFrame, max_budget: int) -> pd.DataFrame:
    """
    Analyzes relationship between property size and value (Chunk 3.2 from Colab).

    Identifies "Size Traps" - properties that appear cheap but are expensive per sqft.

    CACHED: Filtering is deterministic based on budget.

    Args:
        df: Market context dataframe
        max_budget: Maximum acquisition budget

    Returns:
        Filtered dataframe for visualization
    """
    # Filter for relevant range
    deep_dive_data = df[
        (df['price'] <= max_budget) &
        (df['condition'] <= 3)
        ].copy()

    return deep_dive_data


# =============================================================================
# CHUNK 3.3: DENSITY ANALYSIS
# =============================================================================

@st.cache_data(show_spinner="🏘️ Analyzing property density types...")
def chunk_03_03_density_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes profit potential by property density type (Chunk 3.3 from Colab).

    CACHED: Grouping is deterministic.

    Args:
        df: Market context dataframe with 'density_type'

    Returns:
        Summary statistics by density type
    """
    density_stats = df.groupby('density_type').agg({
        'sqft_gap': ['mean', 'median', 'count'],
        'price_gap': ['mean', 'median']
    }).round(2)

    density_stats.columns = ['_'.join(col).strip() for col in density_stats.columns.values]
    density_stats = density_stats.reset_index()

    return density_stats


# =============================================================================
# CHUNK 3.4: VALUE HIERARCHY (TOP ZIPCODES)
# =============================================================================

@st.cache_data(show_spinner="🏆 Identifying premium zones...")
def chunk_03_04_value_hierarchy(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Identifies top zipcodes by price per sqft (Chunk 3.4 from Colab).

    CACHED: Sorting is deterministic.

    Args:
        df: Market context dataframe
        top_n: Number of top zipcodes to return

    Returns:
        Filtered dataframe with top zipcodes
    """
    top_zips = df.groupby('zipcode')['price_per_sqft'].mean().sort_values(ascending=False).head(top_n).index

    data_viz = df[df['zipcode'].isin(top_zips)].copy()

    return data_viz


# =============================================================================
# CHUNK 3.5: CONDITION STRATEGY (DATA-DRIVEN SELECTION)
# =============================================================================

@st.cache_data(show_spinner="🔧 Detecting optimal property conditions...")
def chunk_03_05_condition_strategy(df: pd.DataFrame, max_budget: int) -> Dict:
    """
    Auto-detects optimal property conditions based on data (Chunk 3.5 from Colab).

    Selection criteria:
    1. Median price <= budget OR significant volume under budget
    2. Positive profit potential
    3. Business rule: condition <= 3 (renovation focus)

    CACHED: Detection is deterministic.

    Args:
        df: Market context dataframe
        max_budget: Maximum acquisition budget

    Returns:
        Dictionary with optimal conditions and analysis
    """
    MAX_CONDITION_BUSINESS = 3  # Business rule

    # Aggregate by condition
    condition_analysis = df.groupby('condition').agg({
        'price': 'median',
        'price_gap': 'median',
        'id': 'count'
    }).round(0)
    condition_analysis.columns = ['median_price', 'median_gap', 'count']

    # Categorize conditions
    ideal_conditions = []
    acceptable_conditions = []

    for cond in condition_analysis.index:
        price = condition_analysis.loc[cond, 'median_price']
        gap = condition_analysis.loc[cond, 'median_gap']
        count = condition_analysis.loc[cond, 'count']

        fits_budget = price <= max_budget
        has_profit = gap > 0
        within_business_rule = cond <= MAX_CONDITION_BUSINESS

        if fits_budget and has_profit:
            ideal_conditions.append(int(cond))
        elif within_business_rule and has_profit:
            acceptable_conditions.append(int(cond))

    target_conditions = ideal_conditions + acceptable_conditions
    max_condition = max(target_conditions) if target_conditions else MAX_CONDITION_BUSINESS

    return {
        'ideal': ideal_conditions,
        'acceptable': acceptable_conditions,
        'target': target_conditions,
        'max_condition': max_condition,
        'analysis_table': condition_analysis
    }


# =============================================================================
# CHUNK 3.6: VINTAGE VALUE (DATA-DRIVEN DECADES)
# =============================================================================

@st.cache_data(show_spinner="📅 Detecting optimal construction decades...")
def chunk_03_06_vintage_value(df: pd.DataFrame) -> Dict:
    """
    Auto-detects optimal construction decades (Chunk 3.6 from Colab).

    Selection criteria:
    1. Median price gap > market median
    2. Median price gap > 0 (profitable)

    CACHED: Detection is deterministic.

    Args:
        df: Market context dataframe with 'decade' column

    Returns:
        Dictionary with optimal decades and analysis
    """
    # Ensure decade column exists
    if 'decade' not in df.columns:
        df['decade'] = (df['yr_built'] // 10) * 10

    # Analyze by decade
    decade_performance = df.groupby('decade')['price_gap'].median()
    decade_market_median = decade_performance.median()

    # Detect optimal decades
    optimal_decades = decade_performance[
        (decade_performance > decade_market_median) &
        (decade_performance > 0)
        ].index.tolist()

    # Convert to int for consistency
    optimal_decades = [int(d) for d in optimal_decades]

    return {
        'optimal_decades': optimal_decades,
        'market_median': decade_market_median,
        'decade_performance': decade_performance.to_dict()
    }


# =============================================================================
# CHUNK 3.7: GRADE VALUE (DATA-DRIVEN GRADES)
# =============================================================================

@st.cache_data(show_spinner="🏗️ Detecting optimal building grades...")
def chunk_03_07_grade_value(df: pd.DataFrame) -> Dict:
    """
    Auto-detects optimal building grades (Chunk 3.7 from Colab).

    Selection criteria:
    1. Median price gap > market median
    2. Median price gap > 0 (profitable)
    3. Grade >= 5 (minimum structural quality for renovation)

    CACHED: Detection is deterministic.

    Args:
        df: Market context dataframe

    Returns:
        Dictionary with optimal grades and analysis
    """
    MIN_GRADE_FOR_RENOVATION = 5  # Business rule

    # Analyze by grade
    grade_performance = df.groupby('grade')['price_gap'].median()
    grade_market_median = grade_performance.median()

    # Detect optimal grades
    optimal_grades = grade_performance[
        (grade_performance > grade_market_median) &
        (grade_performance > 0) &
        (grade_performance.index >= MIN_GRADE_FOR_RENOVATION)
        ].index.tolist()

    # Convert to int
    optimal_grades = [int(g) for g in optimal_grades]

    return {
        'optimal_grades': optimal_grades,
        'market_median': grade_market_median,
        'min_grade': MIN_GRADE_FOR_RENOVATION,
        'grade_performance': grade_performance.to_dict()
    }


# =============================================================================
# CHUNK 3.8: CORRELATION ANALYSIS (PEARSON VS SPEARMAN)
# =============================================================================

@st.cache_data(show_spinner="📈 Computing correlations...")
def chunk_03_08_correlation_analysis(df: pd.DataFrame) -> Dict:
    """
    Performs correlation analysis using both Pearson and Spearman methods (Chunk 3.8 from Colab).

    CACHED: Correlation calculations are deterministic.

    Args:
        df: Market context dataframe

    Returns:
        Dictionary with both correlation matrices and comparison
    """
    cols_for_corr = ['price', 'sqft_living', 'grade', 'condition', 'yr_built', 'price_gap']

    # Calculate both methods
    corr_pearson = df[cols_for_corr].corr(method='pearson')
    corr_spearman = df[cols_for_corr].corr(method='spearman')

    # Method comparison for price
    comparison = []
    for col in cols_for_corr:
        if col != 'price':
            p = corr_pearson.loc['price', col]
            s = corr_spearman.loc['price', col]
            diff = abs(p - s)
            comparison.append({
                'variable': col,
                'pearson': p,
                'spearman': s,
                'difference': diff,
                'divergent': diff > 0.05
            })

    return {
        'pearson': corr_pearson,
        'spearman': corr_spearman,
        'comparison': pd.DataFrame(comparison),
        'recommended_method': 'spearman'  # Due to high skewness
    }


# =============================================================================
# CHUNK 3.9: CHI-SQUARE INDEPENDENCE TESTS
# =============================================================================

@st.cache_data(show_spinner="🧪 Running Chi-Square tests...")
def chunk_03_09_chi_square_tests(df: pd.DataFrame) -> Dict:
    """
    Performs Chi-Square independence tests (Chunk 3.9 from Colab).

    Tests relationships between categorical variables.

    CACHED: Statistical tests are deterministic.

    Args:
        df: Market context dataframe

    Returns:
        Dictionary with test results
    """
    # Prepare categorical variables
    df_test = df.copy()

    # Create categories if not exist
    if 'decade' not in df_test.columns:
        df_test['decade'] = (df_test['yr_built'] // 10) * 10

    df_test['condition_cat'] = df_test['condition'].astype(int).astype(str)

    df_test['grade_tier'] = pd.cut(
        df_test['grade'],
        bins=[0, 4, 7, 15],
        labels=['Low (1-4)', 'Mid (5-7)', 'High (8+)']
    )

    df_test['era'] = pd.cut(
        df_test['yr_built'],
        bins=[0, 1945, 1980, 2000, 2020],
        labels=['Pre-1945', '1945-1980', '1980-2000', 'Post-2000']
    )

    # Define tests to run
    tests = [
        ('era', 'condition_cat', 'Construction Era', 'Property Condition'),
        ('grade_tier', 'condition_cat', 'Grade Tier', 'Property Condition'),
        ('density_type', 'condition_cat', 'Density Type', 'Property Condition'),
        ('era', 'grade_tier', 'Construction Era', 'Grade Tier')
    ]

    results = []
    for var1, var2, name1, name2 in tests:
        contingency = pd.crosstab(df_test[var1], df_test[var2])
        chi2, p_value, dof, expected = chi2_contingency(contingency)

        results.append({
            'var1': name1,
            'var2': name2,
            'chi2': chi2,
            'p_value': p_value,
            'dof': dof,
            'significant': p_value < 0.05
        })

    return {
        'results': pd.DataFrame(results),
        'df_with_categories': df_test
    }


# =============================================================================
# CHUNK 3.10: ANOVA - COMPARING GROUP MEANS
# =============================================================================

@st.cache_data(show_spinner="📊 Running ANOVA tests...")
def chunk_03_10_anova_tests(df: pd.DataFrame) -> Dict:
    """
    Performs One-Way ANOVA tests (Chunk 3.10 from Colab).

    Tests if group means differ significantly.

    CACHED: Statistical tests are deterministic.

    Args:
        df: Market context dataframe with categorical columns

    Returns:
        Dictionary with test results
    """
    # Ensure categorical columns exist
    df_test = df.copy()

    if 'grade_tier' not in df_test.columns:
        df_test['grade_tier'] = pd.cut(
            df_test['grade'],
            bins=[0, 4, 7, 15],
            labels=['Low (1-4)', 'Mid (5-7)', 'High (8+)']
        )

    if 'era' not in df_test.columns:
        df_test['era'] = pd.cut(
            df_test['yr_built'],
            bins=[0, 1945, 1980, 2000, 2020],
            labels=['Pre-1945', '1945-1980', '1980-2000', 'Post-2000']
        )

    # Define tests
    tests = [
        ('condition', 'price_gap', 'Property Condition', 'Price Gap'),
        ('era', 'price_gap', 'Construction Era', 'Price Gap'),
        ('grade_tier', 'price_gap', 'Grade Tier', 'Price Gap'),
        ('density_type', 'price_gap', 'Density Type', 'Price Gap'),
        ('grade_tier', 'price', 'Grade Tier', 'Price')
    ]

    results = []
    group_means = {}

    for group_col, value_col, group_name, value_name in tests:
        # Get groups
        groups = df_test[group_col].dropna().unique()
        group_data = [df_test[df_test[group_col] == g][value_col].dropna() for g in groups]

        # Perform ANOVA
        f_stat, p_value = f_oneway(*group_data)

        # Get means
        means = df_test.groupby(group_col, observed=False)[value_col].mean().sort_values(ascending=False).to_dict()

        results.append({
            'group': group_name,
            'measure': value_name,
            'f_stat': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        })

        group_means[f"{group_name}_{value_name}"] = means

    return {
        'results': pd.DataFrame(results),
        'group_means': group_means
    }