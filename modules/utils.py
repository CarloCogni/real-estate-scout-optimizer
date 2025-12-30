"""
Utility functions for formatting, display, and common operations
"""

import pandas as pd
import streamlit as st


# =============================================================================
# FORMATTING FUNCTIONS
# =============================================================================

def format_currency(value: float, compact: bool = False) -> str:
    """
    Format currency values with proper separators.

    Args:
        value: Numeric value
        compact: If True, use K/M notation (e.g., 310K instead of 310,000)

    Returns:
        Formatted string

    Examples:
        >>> format_currency(300000)
        '$300,000'
        >>> format_currency(300000, compact=True)
        '$300K'
        >>> format_currency(1500000, compact=True)
        '$1.5M'
    """
    if compact:
        if abs(value) >= 1_000_000:
            return f"${value / 1_000_000:.1f}M"
        elif abs(value) >= 1_000:
            return f"${value / 1_000:.0f}K"
        else:
            return f"${value:.0f}"
    else:
        return f"${value:,.0f}"


def format_number(value: float, decimals: int = 0) -> str:
    """
    Format numbers with thousand separators.

    Args:
        value: Numeric value
        decimals: Number of decimal places

    Returns:
        Formatted string

    Examples:
        >>> format_number(21628)
        '21,628'
        >>> format_number(1234.567, decimals=2)
        '1,234.57'
    """
    if decimals == 0:
        return f"{value:,.0f}"
    else:
        return f"{value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format percentage values.

    Args:
        value: Numeric value (0.15 = 15%)
        decimals: Number of decimal places

    Returns:
        Formatted string with % symbol

    Examples:
        >>> format_percentage(0.156)
        '15.6%'
        >>> format_percentage(0.156, decimals=0)
        '16%'
    """
    return f"{value * 100:.{decimals}f}%"


def format_sqft(value: float) -> str:
    """
    Format square footage values.

    Args:
        value: Square footage

    Returns:
        Formatted string with unit

    Examples:
        >>> format_sqft(1500)
        '1,500 sqft'
    """
    return f"{value:,.0f} sqft"


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def display_metric_row(metrics: dict, columns: int = 4):
    """
    Display a row of metrics using Streamlit columns.

    Args:
        metrics: Dictionary with keys as labels and values as (value, delta) tuples
        columns: Number of columns to create

    Example:
        >>> display_metric_row({
        ...     "Mean Price": (300000, "+10K"),
        ...     "Median": (250000, None),
        ...     "Count": (1000, None)
        ... }, columns=3)
    """
    cols = st.columns(columns)
    for idx, (label, data) in enumerate(metrics.items()):
        with cols[idx % columns]:
            if isinstance(data, tuple):
                value, delta = data
                st.metric(label, value, delta=delta)
            else:
                st.metric(label, data)


def display_dataframe_with_download(
        df: pd.DataFrame,
        filename: str,
        title: str = None,
        key: str = None
):
    """
    Display a dataframe with download button.

    Args:
        df: DataFrame to display
        filename: Name for downloaded file
        title: Optional title above dataframe
        key: Unique key for download button
    """
    if title:
        st.markdown(f"#### {title}")

    st.dataframe(df, use_container_width=True, hide_index=True)

    # Convert to CSV for download
    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label=f"📥 Download {filename}",
        data=csv,
        file_name=filename,
        mime="text/csv",
        key=key or f"download_{filename}"
    )


# =============================================================================
# DATA HELPERS
# =============================================================================

def create_download_link(df: pd.DataFrame, filename: str, link_text: str = "Download CSV") -> str:
    """
    Create a download link for a dataframe (legacy - use st.download_button instead).

    Args:
        df: DataFrame to download
        filename: Name of file
        link_text: Text for the link

    Returns:
        HTML string for download link
    """
    csv = df.to_csv(index=False)
    import base64
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href


def get_summary_stats(series: pd.Series) -> dict:
    """
    Get comprehensive summary statistics for a series.

    Args:
        series: Pandas Series

    Returns:
        Dictionary with statistics
    """
    return {
        'count': series.count(),
        'mean': series.mean(),
        'median': series.median(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        'q25': series.quantile(0.25),
        'q75': series.quantile(0.75),
        'skew': series.skew(),
        'kurtosis': series.kurtosis()
    }