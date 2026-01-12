"""
Plotly Interactive Visualizations
Enhanced interactive charts for web deployment
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# =============================================================================
# STRATEGIC ZONES SCATTER (PLOTLY VERSION)
# =============================================================================
@st.cache_data(show_spinner=False)
def plot_strategic_zones_interactive(zip_strategy: pd.DataFrame) -> go.Figure:
    """
    Creates INTERACTIVE scatter plot of zipcodes by volume and profit.
    UPDATED: Uses Median logic and adapted scales.
    """
    # 1. Filter: Relax constraint (Same as Colab)
    zip_strategy_filtered = zip_strategy[zip_strategy['opportunity_count'] >= 2].copy()

    # 2. Calculate reference lines (Medians of the FILTERED data)
    # This ensures the red/blue lines match the visible data, same as Colab
    median_profit = zip_strategy_filtered['avg_potential_profit'].median()
    median_count = zip_strategy_filtered['opportunity_count'].median()

    # Create hover text
    zip_strategy_filtered['hover_text'] = zip_strategy_filtered.apply(
        lambda row: (
            f"<b>Zipcode {int(row['zipcode'])}</b><br>"
            f"Opportunities: {int(row['opportunity_count'])}<br>"
            f"Median Profit: ${row['avg_potential_profit']:,.0f}<br>"
            f"Nbhd Avg Price: ${row['zip_avg_price']:,.0f}<br>"
        ),
        axis=1
    )

    fig = go.Figure()

    # Add Scatter Trace
    fig.add_trace(go.Scatter(
        x=zip_strategy_filtered['opportunity_count'],
        y=zip_strategy_filtered['avg_potential_profit'],
        mode='markers',
        marker=dict(
            # ADJUSTED SIZE: Divided by 2000 instead of 8000 to make bubbles visible
            size=zip_strategy_filtered['zip_avg_price'] / 15_000,
            sizemin=8,  # Minimum size to ensure visibility
            color=zip_strategy_filtered['avg_potential_profit'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title=dict(text="Median Profit ($)", side="right"),
                tickformat="$,.0f"
            ),
            line=dict(width=1, color='black')
        ),
        text=zip_strategy_filtered['hover_text'],
        hovertemplate='%{text}<extra></extra>',
        name='Zipcodes'
    ))

    # Reference Lines
    fig.add_hline(y=median_profit, line_dash="dash", line_color="red", opacity=0.6,
                  annotation_text=f"Median Profit: ${median_profit/1000:.0f}K",
                  annotation_position="top right")

    fig.add_vline(x=median_count, line_dash="dash", line_color="blue", opacity=0.6,
                  annotation_text=f"Med Vol: {median_count:.0f}",
                  annotation_position="top right")

    # Annotate Top 5
    top_5 = zip_strategy_filtered.nlargest(5, 'avg_potential_profit')
    for idx, row in top_5.iterrows():
        fig.add_annotation(
            x=row['opportunity_count'],
            y=row['avg_potential_profit'],
            text=f"{int(row['zipcode'])}",
            showarrow=True, arrowhead=2, arrowsize=1, arrowcolor="gray",
            ax=20, ay=-20, bgcolor="yellow", opacity=0.8
        )

    # Layout
    fig.update_layout(
        title={
            'text': 'STRATEGIC ZONES: Profit vs. Volume (Median Adjusted)',
            'x': 0.5, 'xanchor': 'center'
        },
        xaxis_title="Volume of Opportunities (Number of Houses)",
        yaxis_title="Median Potential Profit per House",
        yaxis=dict(tickformat="$,.0f", dtick=25000), # 25k ticks matches Matplotlib
        xaxis=dict(dtick=2),                         # 2 unit ticks matches Matplotlib
        hovermode='closest',
        template='plotly_white',
        height=650
    )

    return fig
