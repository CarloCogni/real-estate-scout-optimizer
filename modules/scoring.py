"""
Smart Score Algorithm Module
Chunks 4-5 from Colab notebook

This module implements the Smart Scout Score algorithm:
1. Efficiency Gate (trap filter)
2. Weighted scoring system (80% financial, 10% structural, 10% vintage)
3. Validation metrics
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Tuple


# =============================================================================
# CHUNK 4: STRATEGY IMPLEMENTATION (FILTERING)
# =============================================================================

@st.cache_data(show_spinner="🎯 Applying strategy filters...")
def chunk_04_strategy_filter(
        df: pd.DataFrame,
        max_budget: int,
        target_conditions: list
) -> pd.DataFrame:
    """
    Applies initial strategy filters (Chunk 4 from Colab).

    Filters:
    1. Price <= max_budget
    2. Condition <= max(target_conditions)

    CACHED: Filtering is deterministic.

    Args:
        df: Market context dataframe
        max_budget: Maximum acquisition budget
        target_conditions: List of acceptable conditions

    Returns:
        Filtered dataframe (scout candidate list)
    """
    max_condition = max(target_conditions) if target_conditions else 3

    df_scout = df[
        (df['price'] <= max_budget) &
        (df['condition'] <= max_condition)
        ].copy()

    return df_scout


# =============================================================================
# CHUNK 5: INTELLIGENCE ENGINE (SMART SCORE)
# =============================================================================

@st.cache_data(show_spinner="🧠 Computing Smart Scores...")
def chunk_05_smart_score(
        df_scout: pd.DataFrame,
        df_full: pd.DataFrame,  # ✅ ADD: Full dataset for confidence check
        optimal_grades: list,
        optimal_decades: list,
        min_zipcode_samples: int = 15
) -> Tuple[pd.DataFrame, Dict]:
    """
    Calculates Smart Score with efficiency gate (Chunk 5 from Colab).

    Algorithm:
    1. Confidence Filter: Remove zipcodes with < 15 samples IN FULL DATASET
    2. Efficiency Gate: Remove properties with negative sqft_gap (traps)
    3. Score Calculation:
       - Financial (80%): Normalized price_gap
       - Structural (10%): Bonus for optimal grades
       - Vintage (10%): Bonus for optimal decades

    CACHED: Scoring is deterministic based on parameters.

    Args:
        df_scout: Filtered scout candidates
        df_full: Full cleaned dataset (for confidence check)
        optimal_grades: List of optimal building grades
        optimal_decades: List of optimal construction decades
        min_zipcode_samples: Minimum samples per zipcode for reliability

    Returns:
        Tuple of (scored_dataframe, audit_dict)
    """
    audit = {}

    # A. Confidence Filter (Statistical Safety)
    # ✅ FIX: Count on FULL dataset, not filtered scout list
    zip_counts = df_full['zipcode'].value_counts()
    reliable_zips = zip_counts[zip_counts >= min_zipcode_samples].index

    df_ranked = df_scout[df_scout['zipcode'].isin(reliable_zips)].copy()

    audit['candidates_after_confidence_filter'] = len(df_ranked)
    audit['zipcodes_excluded'] = len(df_scout['zipcode'].unique()) - len(reliable_zips)

    # B. EFFICIENCY GATE (Pass/Fail) - Critical Trap Filter
    traps_filter = df_ranked['sqft_gap'] < 0
    traps_count = traps_filter.sum()

    df_ranked = df_ranked[~traps_filter].copy()

    audit['size_traps_blocked'] = traps_count
    audit['candidates_after_trap_filter'] = len(df_ranked)

    # C. Calculate Scores (0-100 scale)

    # 1. Financial Score (80% weight) - Based on Price Gap
    max_gap = df_ranked['price_gap'].max()
    df_ranked['score_finance'] = (df_ranked['price_gap'] / max_gap) * 100

    # 2. Structural Score (10% weight) - Bonus for Optimal Grades
    df_ranked['score_structure'] = df_ranked['grade'].apply(
        lambda x: 100 if x in optimal_grades else 0
    )

    # 3. Vintage Score (10% weight) - Bonus for Optimal Decades
    df_ranked['score_vintage'] = df_ranked['yr_built'].apply(
        lambda x: 100 if (x // 10) * 10 in optimal_decades else 0
    )

    # D. Final Weighted Score
    df_ranked['SMART_SCORE'] = (
            (df_ranked['score_finance'] * 0.8) +
            (df_ranked['score_structure'] * 0.1) +
            (df_ranked['score_vintage'] * 0.1)
    )

    # E. Sort by score (descending)
    df_ranked = df_ranked.sort_values(by='SMART_SCORE', ascending=False)

    # Audit stats
    audit['max_score'] = df_ranked['SMART_SCORE'].max()
    audit['mean_score'] = df_ranked['SMART_SCORE'].mean()
    audit['median_score'] = df_ranked['SMART_SCORE'].median()
    audit['top_10_mean_gap'] = df_ranked.head(10)['price_gap'].mean()

    return df_ranked, audit

# =============================================================================
# CHUNK 5B: SMART SCORE VALIDATION
# =============================================================================

@st.cache_data(show_spinner="✅ Validating Smart Score...")
def chunk_05b_validate_score(df_ranked: pd.DataFrame) -> Dict:
    """
    Validates Smart Score performance (Chunk 5b from Colab).

    Creates score tiers and compares metrics to ensure algorithm works correctly.

    CACHED: Validation is deterministic.

    Args:
        df_ranked: Dataframe with SMART_SCORE column

    Returns:
        Dictionary with validation metrics
    """
    # Create score tiers
    df_validation = df_ranked.copy()
    df_validation['score_tier'] = pd.cut(
        df_validation['SMART_SCORE'],
        bins=[0, 40, 60, 100],
        labels=['Low (<40)', 'Mid (40-60)', 'High (>60)']
    )

    # Compare metrics across tiers
    validation_table = df_validation.groupby('score_tier', observed=False).agg({
        'price_gap': 'median',
        'sqft_gap': 'median',
        'price_per_sqft': 'median',
        'id': 'count'
    }).round(2)

    validation_table.columns = ['median_price_gap', 'median_sqft_gap', 'median_price_per_sqft', 'count']

    # Calculate improvement (High vs Low tier)
    if 'High (>60)' in validation_table.index and 'Low (<40)' in validation_table.index:
        high_gap = validation_table.loc['High (>60)', 'median_price_gap']
        low_gap = validation_table.loc['Low (<40)', 'median_price_gap']
        improvement_pct = ((high_gap - low_gap) / low_gap) * 100 if low_gap > 0 else 0
    else:
        improvement_pct = 0

    return {
        'validation_table': validation_table,
        'improvement_pct': improvement_pct,
        'tier_distribution': df_validation['score_tier'].value_counts().to_dict()
    }


# =============================================================================
# CHUNK 6: GEOGRAPHIC OPTIMIZATION (STRATEGIC ZONES)
# =============================================================================

@st.cache_data(show_spinner="🗺️ Identifying strategic zones...")
def chunk_06_strategic_zones(df_ranked: pd.DataFrame, min_properties: int = 5) -> pd.DataFrame:
    """
    Identifies best zipcodes for geographic targeting (Chunk 6 from Colab).

    Aggregates opportunities by zipcode to find high-volume, high-profit zones.

    CACHED: Aggregation is deterministic.

    Args:
        df_ranked: Scored dataframe
        min_properties: Minimum properties per zipcode to include

    Returns:
        Zipcode-level summary with strategic metrics
    """
    zip_strategy = df_ranked.groupby('zipcode').agg({
        'id': 'count',
        'price_gap': 'mean',
        'zip_avg_price': 'mean',
        'SMART_SCORE': 'mean'
    }).reset_index()

    zip_strategy.rename(columns={
        'id': 'opportunity_count',
        'price_gap': 'avg_potential_profit',
        'SMART_SCORE': 'avg_smart_score'
    }, inplace=True)

    # Filter out low-volume zipcodes
    zip_strategy_filtered = zip_strategy[
        zip_strategy['opportunity_count'] >= min_properties
        ].copy()

    # Sort by average profit (descending)
    zip_strategy_filtered = zip_strategy_filtered.sort_values(
        by='avg_potential_profit',
        ascending=False
    )

    return zip_strategy_filtered


# =============================================================================
# EXPORT PREPARATION
# =============================================================================

@st.cache_data(show_spinner="📦 Preparing export files...")
def prepare_export_files(
        df_clean: pd.DataFrame,
        df_ranked: pd.DataFrame,
        target_ids: list = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepares export files for download.

    Creates:
    1. Action List (Top deals with key metrics)
    2. Full Dataset (All properties with target flag and scores)

    CACHED: Export preparation is deterministic.

    Args:
        df_clean: Full cleaned dataset
        df_ranked: Scored targets
        target_ids: List of target property IDs (optional)

    Returns:
        Tuple of (action_list_df, full_dataset_df)
    """
    # 1. ACTION LIST (Top deals only)
    action_columns = [
        'id', 'SMART_SCORE', 'price', 'price_gap', 'sqft_gap',
        'zipcode', 'grade', 'condition', 'yr_built', 'sqft_living',
        'lat', 'long'
    ]

    action_list = df_ranked[action_columns].copy()

    # 2. FULL DATASET (with target flag)
    export_df = df_clean.copy()

    # Mark targets
    if target_ids is None:
        target_ids = df_ranked['id'].unique()

    export_df['is_scout_target'] = export_df['id'].apply(
        lambda x: 'Target' if x in target_ids else 'Non-Target'
    )

    # Merge Smart Score (only targets will have scores)
    export_df = export_df.merge(
        df_ranked[['id', 'SMART_SCORE']],
        on='id',
        how='left'
    )
    export_df['SMART_SCORE'] = export_df['SMART_SCORE'].fillna(0)

    return action_list, export_df


# =============================================================================
# UTILITY: GET TOP DEALS
# =============================================================================

def get_top_deals(df_ranked: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Returns top N deals sorted by Smart Score.

    NOT CACHED: Simple filtering operation.

    Args:
        df_ranked: Scored dataframe
        n: Number of top deals to return

    Returns:
        Top N properties
    """
    return df_ranked.head(n).copy()