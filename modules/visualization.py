"""
Visualization Module
Chart generation and interactive mapping

This module contains all visualization functions from the Colab notebook.
Each function returns matplotlib/seaborn figures or Folium maps.
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import folium
from folium import plugins
from folium.plugins import MarkerCluster, HeatMap
from typing import Tuple, Dict, Any
from scipy.stats import trim_mean

# Set default style for all plots
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


# =============================================================================
# CHUNK 3.1: MARKET DISTRIBUTION PLOTS
# =============================================================================

@st.cache_data(show_spinner=False)
def plot_market_distribution(df: pd.DataFrame, stats: dict) -> plt.Figure:
    """
    Creates dual plot: Price distribution + Opportunity distribution (Chunk 3.1).

    CACHED: Plot generation is deterministic.

    Args:
        df: Market context dataframe
        stats: Statistics dict from chunk_03_01_univariate_analysis

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Price Distribution
    sns.histplot(df['price'], kde=True, bins=50, color='blue', ax=axes[0])
    axes[0].set_title(
        f"Market Price Distribution\nSkew: {stats['skewness']:.2f} | Kurtosis: {stats['kurtosis']:.2f}",
        fontsize=14,
        fontweight='bold'
    )
    axes[0].set_xlabel('Price ($)')
    axes[0].set_xlim(0, 2000000)
    axes[0].xaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))

    # Plot 2: Opportunity Distribution
    sns.histplot(df['price_gap'], kde=True, bins=50, color='green', ax=axes[1])
    axes[1].set_title('Opportunity Distribution (Profit Potential)', fontsize=14, fontweight='bold')
    axes[1].axvline(0, color='red', linestyle='--', label='Zero Profit', linewidth=2)
    axes[1].set_xlabel('Price Gap ($)')
    axes[1].xaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
    axes[1].legend()

    plt.tight_layout()
    return fig


# =============================================================================
# CHUNK 3.2: SIZE TRAP ANALYSIS (JOINT PLOT)
# =============================================================================

@st.cache_data(show_spinner=False)
def plot_size_trap_analysis(deep_dive_data: pd.DataFrame) -> plt.Figure:
    """
    Creates joint plot showing size vs value relationship (Chunk 3.2).

    CACHED: Plot generation is deterministic.

    Args:
        deep_dive_data: Filtered dataframe from chunk_03_02_size_trap_analysis

    Returns:
        Matplotlib figure
    """
    from matplotlib.gridspec import GridSpec

    # ✅ FIX: Reduced figure size and set explicit DPI to avoid decompression bomb
    fig = plt.figure(figsize=(12, 9), dpi=100)  # DPI 100 instead of default 150+
    gs = GridSpec(4, 4, hspace=0.05, wspace=0.05, figure=fig)

    # Main scatter plot
    ax_main = fig.add_subplot(gs[1:, :-1])

    # Marginal plots
    ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)

    # Main scatter with seaborn
    scatter = sns.scatterplot(
        data=deep_dive_data,
        x='sqft_living',
        y='price_gap',
        hue='sqft_gap',
        size='sqft_lot',
        sizes=(15, 150),  # ✅ Slightly smaller bubbles
        palette='RdYlGn',
        alpha=0.6,
        edgecolor='black',
        linewidth=0.3,
        ax=ax_main,
        legend='brief'
    )

    ax_main.axhline(0, color='grey', linestyle='--', linewidth=1.5)
    ax_main.set_xlabel('Living Space (sqft)', fontsize=11, fontweight='bold')
    ax_main.set_ylabel('Price Gap ($)', fontsize=11, fontweight='bold')
    ax_main.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
    ax_main.grid(True, alpha=0.3)

    # ✅ FIX: Simple legend positioning without complex customization
    handles, labels = ax_main.get_legend_handles_labels()

    # Format labels for better readability
    formatted_labels = []
    for label in labels:
        if label == 'sqft_gap':
            formatted_labels.append('Efficiency ($/sqft gap)')
        elif label == 'sqft_lot':
            formatted_labels.append('Lot Size (sqft)')
        else:
            # Try to format numbers with commas
            try:
                val = float(label)
                if abs(val) >= 1000:
                    formatted_labels.append(f'{val:,.0f}')
                else:
                    formatted_labels.append(f'{val:.0f}')
            except ValueError:
                formatted_labels.append(label)

    # Create legend with formatted labels
    legend = ax_main.legend(
        handles=handles,
        labels=formatted_labels,
        loc='upper right',
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        edgecolor='black',
        title='Metrics Key',
        title_fontsize=9,
        ncol=1
    )

    # Top margin: sqft distribution
    ax_top.hist(deep_dive_data['sqft_living'], bins=30, color='steelblue',
                alpha=0.6, edgecolor='black', linewidth=0.5)
    ax_top.set_ylabel('Count', fontsize=9)
    ax_top.tick_params(labelbottom=False, labelsize=8)
    ax_top.grid(True, alpha=0.3, axis='y')

    # Right margin: price gap distribution
    ax_right.hist(deep_dive_data['price_gap'], bins=30, orientation='horizontal',
                  color='steelblue', alpha=0.6, edgecolor='black', linewidth=0.5)
    ax_right.set_xlabel('Count', fontsize=9)
    ax_right.tick_params(labelleft=False, labelsize=8)
    ax_right.grid(True, alpha=0.3, axis='x')

    # Title
    fig.suptitle(
        'THE TRUTH CHECK: Are High-Gap Houses Just Tiny?\n(with Marginal Distributions)',
        fontsize=14,
        fontweight='bold',
        y=0.97
    )

    # Text annotation box
    text_box = "GREEN = TRUE DEALS\n(Cheaper per sqft)\n\nRED = SIZE TRAPS\n(Expensive per sqft!)"
    ax_main.text(
        0.02, 0.98,
        text_box,
        transform=ax_main.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.9, edgecolor='black', linewidth=1),
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='left'
    )

    # ✅ FIX: Use subplots_adjust instead of tight_layout to avoid warnings
    plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.08)

    return fig

# =============================================================================
# CHUNK 3.3: DENSITY ANALYSIS
# =============================================================================

@st.cache_data(show_spinner=False)
def plot_density_analysis(df: pd.DataFrame) -> plt.Figure:
    """
    Creates boxplot of profit by density type (Chunk 3.3).

    CACHED: Plot generation is deterministic.

    Args:
        df: Market context dataframe

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Order for display
    order = [
        'Condo/Townhouse (High Density)',
        'Small Yard (Suburban)',
        'Large Lot (Villa/Estate)'
    ]

    sns.boxplot(
        data=df,
        x='density_type',
        y='sqft_gap',
        hue='density_type',
        palette='coolwarm',
        order=order,
        legend=False,
        ax=ax
    )

    ax.axhline(0, color='red', linestyle='--', label='Avg Market Value', linewidth=2)
    ax.set_title('Density Analysis: Do Condos Offer Less Value per Sqft?', fontsize=14, fontweight='bold')
    ax.set_ylabel('Profit Gap per Sqft ($)', fontsize=12)
    ax.set_xlabel('Property Type', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    return fig


# =============================================================================
# CHUNK 3.4: VALUE HIERARCHY
# =============================================================================

@st.cache_data(show_spinner=False)
def plot_value_hierarchy(data_viz: pd.DataFrame) -> plt.Figure:
    """
    Creates boxplot of top zipcodes by price per sqft (Chunk 3.4).

    CACHED: Plot generation is deterministic.

    Args:
        data_viz: Filtered dataframe from chunk_03_04_value_hierarchy

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # Get order by median
    order = data_viz.groupby('zipcode')['price_per_sqft'].median().sort_values(ascending=False).index

    sns.boxplot(
        data=data_viz,
        x='zipcode',
        y='price_per_sqft',
        order=order,
        palette='magma',
        hue="zipcode",
        ax=ax,
    )

    ax.set_title('THE VALUE HIERARCHY: Top 15 Zipcodes by Price per Sqft', fontsize=16, fontweight='bold')
    ax.set_ylabel('Price per Sqft ($)', fontsize=12)
    ax.set_xlabel('Zipcode', fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


# =============================================================================
# CHUNK 3.5: CONDITION STRATEGY
# =============================================================================

@st.cache_data(show_spinner=False)
def plot_condition_strategy(df: pd.DataFrame, max_budget: int) -> plt.Figure:
    """
    Creates dual panel: Cost vs Opportunity by condition (Chunk 3.5).

    CACHED: Plot generation is deterministic.

    Args:
        df: Market context dataframe
        max_budget: Budget line for reference

    Returns:
        Matplotlib figure
    """
    y_limit_price = df['price'].quantile(0.95)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Acquisition Cost
    sns.stripplot(
        data=df, x='condition', y='price',
        alpha=0.2, jitter=True, color='darkblue', size=3, ax=axes[0]
    )
    sns.boxplot(
        data=df, x='condition', y='price',
        hue='condition', palette='RdYlGn', legend=False,
        showfliers=False, ax=axes[0],
        boxprops=dict(alpha=0.6), width=0.5
    )
    axes[0].axhline(max_budget, color='red', linestyle='--', linewidth=2, label=f'${max_budget / 1000:.0f}K Budget')
    axes[0].set_ylim(0, y_limit_price)
    axes[0].yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
    axes[0].set_title('Acquisition Cost by Condition\n(Each dot = 1 property)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Price ($)')
    axes[0].set_xlabel('Condition')
    axes[0].legend(loc='upper right')

    # Panel 2: Profit Potential
    sns.boxplot(
        data=df, x='condition', y='price_gap',
        hue='condition', palette='RdYlGn', legend=False,
        showfliers=False, ax=axes[1]
    )
    axes[1].axhline(0, color='grey', linestyle='--', linewidth=1.5, alpha=0.7, label='Break-even')
    axes[1].yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
    axes[1].set_title('Profit Potential by Condition\n(Price Gap)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Price Gap ($)')
    axes[1].set_xlabel('Condition')
    axes[1].legend(loc='upper right')

    plt.suptitle('Condition Analysis: Cost vs. Opportunity', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# CHUNK 3.6: VINTAGE VALUE
# =============================================================================

@st.cache_data(show_spinner=False)
def plot_vintage_value(df: pd.DataFrame, optimal_decades: list, market_median: float) -> plt.Figure:
    """
    Creates line plot of profit by construction decade (Chunk 3.6).
    """
    # 1. CLEANING: Filter out bad years (Year 0 or very old outliers)
    # This prevents the "long diagonal line" from year 0 to 1900
    df_viz = df[df['yr_built'] >= 1900].copy()

    # 2. Create Decade Column
    df_viz['decade'] = (df_viz['yr_built'] // 10) * 10

    fig, ax = plt.subplots(figsize=(12, 6))

    # 3. Calculate median per decade on the CLEANED data
    decade_median = df_viz.groupby('decade')['price_gap'].median().reset_index()

    # Plot Line
    ax.plot(decade_median['decade'], decade_median['price_gap'],
            marker='o', color='purple', linewidth=2, markersize=8)

    # Highlight optimal decades
    for decade in optimal_decades:
        # Only highlight if within visible range
        if decade >= 1900:
            ax.axvspan(decade, decade + 10, alpha=0.2, color='gold', linewidth=0)

    # Reference lines
    ax.axhline(0, color='red', linestyle='--', alpha=0.5, label='Break-even (Zero Profit)')
    ax.axhline(market_median, color='blue', linestyle='--', alpha=0.7,
               label=f'Market Median: ${market_median:,.0f}')

    # 4. Set Intelligent X-Axis Limits
    min_decade = 1900  # Force start at 1900
    max_decade = df_viz['decade'].max()
    ax.set_xlim(min_decade - 5, max_decade + 15)

    # Formatting
    ax.set_title(f'Vintage Value: Optimal Decades Highlighted\nOptimal: {optimal_decades}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Construction Decade', fontsize=12)
    ax.set_ylabel('Median Price Gap ($)', fontsize=12)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))

    # Add legend
    ax.legend(loc='lower left')
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    return fig

# =============================================================================
# CHUNK 3.7: GRADE VALUE
# =============================================================================

@st.cache_data(show_spinner=False)
def plot_grade_value(df: pd.DataFrame, optimal_grades: list, market_median: float) -> plt.Figure:
    """
    Creates bar plot of profit by building grade (Chunk 3.7).
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    grade_performance = df.groupby('grade')['price_gap'].median()

    colors = ['gold' if g in optimal_grades else 'steelblue' for g in grade_performance.index]

    bars = ax.bar(
        grade_performance.index.astype(str),
        grade_performance.values,
        color=colors,
        edgecolor='black',
        linewidth=1.5
    )

    ax.set_title(f'Grade Value: Optimal Grades = {optimal_grades}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Building Grade', fontsize=12)
    ax.set_ylabel('Median Price Gap ($)', fontsize=12)

    # --- UPDATE: FIX Y-AXIS TICKS ---
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(250000))  # Tacche ogni 250k
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))

    # Linee di riferimento
    ax.axhline(market_median, color='red', linestyle='--', linewidth=2,
               label=f'Market Median: ${market_median:,.0f}')
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    ax.legend(loc='upper right')  # Spostato legenda per non coprire le barre
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


# =============================================================================
# CHUNK 3.8: CORRELATION HEATMAPS
# =============================================================================

@st.cache_data(show_spinner=False)
def plot_correlation_heatmaps(corr_pearson: pd.DataFrame, corr_spearman: pd.DataFrame) -> plt.Figure:
    """
    Creates dual heatmap: Pearson vs Spearman correlation (Chunk 3.8).

    CACHED: Plot generation is deterministic.

    Args:
        corr_pearson: Pearson correlation matrix
        corr_spearman: Spearman correlation matrix

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sns.heatmap(
        corr_pearson, annot=True, fmt=".2f", cmap='coolwarm',
        vmin=-1, vmax=1, linewidths=0.5, ax=axes[0],
        cbar_kws={'label': 'Correlation'}
    )
    axes[0].set_title('Pearson Correlation\n(Assumes linearity, sensitive to outliers)', fontsize=12)

    sns.heatmap(
        corr_spearman, annot=True, fmt=".2f", cmap='coolwarm',
        vmin=-1, vmax=1, linewidths=0.5, ax=axes[1],
        cbar_kws={'label': 'Correlation'}
    )
    axes[1].set_title('Spearman Correlation\n(Rank-based, robust to outliers) - RECOMMENDED', fontsize=12)
    plt.suptitle('Correlation Method Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# CHUNK 3.10: ANOVA VISUALIZATION
# =============================================================================

@st.cache_data(show_spinner=False)
def plot_anova_condition(df: pd.DataFrame) -> plt.Figure:
    """
    Creates ANOVA visualization for condition effect (Chunk 3.10).

    CACHED: Plot generation is deterministic.

    Args:
        df: Market context dataframe with categorical columns

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Boxplot
    sns.boxplot(
        data=df, x='condition', y='price_gap',
        hue='condition', palette='RdYlGn', legend=False, ax=axes[0]
    )
    axes[0].axhline(0, color='grey', linestyle='--', alpha=0.5)
    axes[0].yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
    axes[0].set_title('Price Gap Distribution by Condition', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Price Gap ($)')
    axes[0].set_xlabel('Condition')

    # Plot 2: Mean with CI
    sns.barplot(
        data=df, x='condition', y='price_gap',
        hue='condition', palette='RdYlGn', legend=False,
        errorbar='ci', ax=axes[1]
    )
    axes[1].axhline(0, color='grey', linestyle='--', alpha=0.5)
    axes[1].yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
    axes[1].set_title('Mean Price Gap by Condition\n(with 95% Confidence Interval)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Mean Price Gap ($)')
    axes[1].set_xlabel('Condition')

    plt.suptitle('ANOVA Visualization: Condition Effect on Profit Potential', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# CHUNK 5B: SCORE VALIDATION PLOTS
# =============================================================================

@st.cache_data(show_spinner=False)
def plot_score_validation(df_validation: pd.DataFrame) -> plt.Figure:
    """
    Creates validation plots for Smart Score (Chunk 5b).

    CACHED: Plot generation is deterministic.

    Args:
        df_validation: Dataframe with score_tier column

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Price Gap by Tier
    sns.boxplot(data=df_validation, x='score_tier', y='price_gap', hue='score_tier', palette='Greens', legend=False,
                ax=axes[0])
    axes[0].set_title('Price Gap by Score Tier', fontsize=12, fontweight='bold')
    axes[0].yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
    axes[0].set_xlabel('Score Tier')
    axes[0].set_ylabel('Price Gap ($)')

    # Plot 2: Efficiency by Tier
    sns.boxplot(data=df_validation, x='score_tier', y='sqft_gap', hue='score_tier', palette='Blues', legend=False,
                ax=axes[1])

    axes[1].set_title('Efficiency ($/sqft Gap) by Score Tier', fontsize=12, fontweight='bold')
    axes[1].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Score Tier')
    axes[1].set_ylabel('$/sqft Gap')

    plt.suptitle('Smart Score Validation: Do Higher Scores = Better Deals?', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# CHUNK 6: STRATEGIC ZONES SCATTER
# =============================================================================

# =============================================================================
# CHUNK 6: STRATEGIC ZONES SCATTER
# =============================================================================

# Add this helper function at the MODULE LEVEL (top of file, after imports)
def _format_thousands(x, pos):
    """Helper function for axis formatting (must be picklable)."""
    return f'${x / 1000:.0f}K'


# =============================================================================
# CHUNK 6: STRATEGIC ZONES SCATTER (STATIC)
# =============================================================================
# --- 1. GLOBAL HELPER FUNCTION (MUST BE HERE TO FIX PICKLE ERROR) ---
def _format_thousands_axis(x, pos):
    """Helper for axis formatting, defined globally for pickling."""
    return f'${x/1000:.0f}K'

@st.cache_data(show_spinner=False)
def plot_strategic_zones(zip_strategy: pd.DataFrame) -> plt.Figure:
    """
    Creates scatter plot of zipcodes by volume and profit (Chunk 6).
    UPDATED: Uses Median logic and adapted scales.
    """
    # 1. Filter: Relax constraint to show more zones
    zip_strategy_filtered = zip_strategy[zip_strategy['opportunity_count'] >= 2].copy()

    fig, ax = plt.subplots(figsize=(14, 10))

    # Create scatter
    scatter = ax.scatter(
        zip_strategy_filtered['opportunity_count'],
        zip_strategy_filtered['avg_potential_profit'],  # This column actually holds the MEDIAN now
        s=zip_strategy_filtered['zip_avg_price'] / 1000,
        c=zip_strategy_filtered['avg_potential_profit'],
        cmap='viridis',
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )

    # Reference lines (Median-based)
    median_profit = zip_strategy_filtered['avg_potential_profit'].median()
    median_vol = zip_strategy_filtered['opportunity_count'].median()

    ax.axhline(median_profit, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax.text(zip_strategy_filtered['opportunity_count'].max(), median_profit + 2000,
            f' Median Profit: ${median_profit / 1000:.0f}K',
            color='red', fontsize=10, fontweight='bold', ha='right')

    ax.axvline(median_vol, color='blue', linestyle='--', alpha=0.5, linewidth=2)
    ax.text(median_vol + 0.1, zip_strategy_filtered['avg_potential_profit'].max(),
            f' Med Vol: {median_vol:.0f}',
            color='blue', fontsize=10, fontweight='bold', rotation=90, va='top')

    # Label top 5 zipcodes
    top_5 = zip_strategy_filtered.nlargest(5, 'avg_potential_profit')
    for idx, row in top_5.iterrows():
        ax.annotate(
            f"{int(row['zipcode'])}",
            (row['opportunity_count'], row['avg_potential_profit']),
            xytext=(10, 10), textcoords='offset points',
            fontweight='bold', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.5)
        )

    # Formatting
    ax.set_title('STRATEGIC ZONES: Profit vs. Volume (Median Adjusted)', fontsize=18, fontweight='bold')
    ax.set_xlabel('Volume of Opportunities (Number of Houses)', fontsize=12)
    ax.set_ylabel('Median Potential Profit per House', fontsize=12)

    # Then inside plot_strategic_zones:
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_format_thousands_axis))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(25000))  # 25k ticks
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))  # 2 unit ticks

    ax.grid(True, linestyle='--', alpha=0.3)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Median Profit ($)', rotation=270, labelpad=20)

    plt.tight_layout()
    return fig

# =============================================================================
# CHUNK 6B: FOLIUM INTERACTIVE MAP
# =============================================================================

@st.cache_data(show_spinner=False)
def create_interactive_map(top_targets: pd.DataFrame, _center_coords: list = None) -> folium.Map:
    """
    Creates interactive Folium map with clusters and heatmap (Chunk 6b).

    CACHED: Map generation is deterministic (note _center_coords prefix for unhashable param).

    Args:
        top_targets: Top scored properties with lat/long
        _center_coords: [lat, long] for map center (optional)

    Returns:
        Folium Map object
    """
    # Calculate center if not provided
    if _center_coords is None:
        center = [top_targets['lat'].mean(), top_targets['long'].mean()]
    else:
        center = _center_coords

    # Base map
    m = folium.Map(location=center, zoom_start=11, tiles='OpenStreetMap')

    # Layer 1: Heatmap
    heat_data = top_targets[['lat', 'long']].values.tolist()
    hm_layer = folium.FeatureGroup(name="📡 Density Radar")
    HeatMap(
        heat_data,
        radius=20,
        blur=25,
        gradient={0.4: '#3b82f6', 0.65: '#8b5cf6', 1: '#581c87'}
    ).add_to(hm_layer)
    hm_layer.add_to(m)

    # Layer 2: Clusters by score tier
    c_green = MarkerCluster(name="🟢 Top Tier (>60)").add_to(m)
    c_orange = MarkerCluster(name="🟡 Mid Tier (40-60)").add_to(m)
    c_red = MarkerCluster(name="🔴 Standard (<40)").add_to(m)

    def get_cluster_color(score):
        if score > 60:
            return c_green, '#00aa00'
        elif score > 40:
            return c_orange, '#ffaa00'
        else:
            return c_red, '#d9534f'

    # Add markers
    for idx, row in top_targets.iterrows():
        cluster, color = get_cluster_color(row['SMART_SCORE'])

        # Color for efficiency
        sqft_color = '#008800' if row['sqft_gap'] > 0 else '#cc0000'

        # Radius based on profit
        radius = (row['price_gap'] / 60000) + 4

        # HTML Popup
        popup_html = f"""
        <div style="width: 260px; font-family: 'Segoe UI', sans-serif; font-size:12px;">
            <div style="background-color:{color}; color:white; padding:5px; border-radius:4px 4px 0 0;">
                <h4 style="margin:0;">House #{row['id']}</h4>
                <small>Score: <b>{row['SMART_SCORE']:.1f}</b> | {row.get('density_type', 'N/A')}</small>
            </div>
            <table style="width:100%; margin-top:5px; border-collapse: collapse;">
                <tr style="border-bottom:1px solid #eee;">
                    <td style="color:#666;">Listing Price:</td>
                    <td style="text-align:right;"><b>${row['price']:,.0f}</b></td>
                </tr>
                <tr style="border-bottom:1px solid #eee; background-color:#f9fdf9;">
                    <td style="color:#666;">Profit Gap:</td>
                    <td style="text-align:right; color:{color}; font-size:13px;"><b>+${row['price_gap']:,.0f}</b></td>
                </tr>
            </table>
            <div style="margin-top:8px; background-color:#f0f0f0; padding:5px; border-radius:4px;">
                <b style="color:#333;">Efficiency Check ($/sqft):</b>
                <table style="width:100%; font-size:11px;">
                    <tr>
                        <td>This House:</td>
                        <td style="text-align:right; color:{sqft_color};"><b>${row['price_per_sqft']:.0f}/ft²</b></td>
                    </tr>
                    <tr>
                        <td>Nbhd Avg:</td>
                        <td style="text-align:right;">${row['zip_avg_price_sqft']:.0f}/ft²</td>
                    </tr>
                </table>
            </div>
            <div style="margin-top:5px; font-size:11px; color:#555;">
                {row.get('bedrooms', 'N/A')} Bed | {row.get('bathrooms', 'N/A'):.0f} Bath | {row['sqft_living']:,} ft² | Grade {row['grade']}
            </div>
        </div>
        """

        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Score: {row['SMART_SCORE']:.0f} | Profit: ${row['price_gap']:,.0f}"
        ).add_to(cluster)

    # Layer control
    folium.LayerControl().add_to(m)

    # Legend
    legend_html = '''
    <div style="position: fixed;
    bottom: 30px; left: 30px; width: 180px; height: 160px;
    border:2px solid #999; z-index:9999; font-size:12px;
    background-color:white; opacity:0.95; padding: 10px; border-radius: 8px;">
    <b>Strategy Map</b><br>
    <small>Density & Efficiency</small>
    <hr style="margin:5px 0;">
    <span style="color:#00aa00;">●</span> Top Tier (>60)<br>
    <span style="color:#ffaa00;">●</span> Mid Tier (40-60)<br>
    <span style="color:#d9534f;">●</span> Standard (<40)<br>
    <hr style="margin:5px 0;">
    <small><i>Click markers for details</i></small>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    return m

