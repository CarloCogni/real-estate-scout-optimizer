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

    Plotly version with hover tooltips and zoom capabilities.

    Args:
        zip_strategy: Zipcode aggregation from chunk_06_strategic_zones

    Returns:
        Plotly Figure object
    """

    zip_strategy = zip_strategy.copy()

    # Calculate reference lines
    avg_profit = zip_strategy['avg_potential_profit'].mean()
    avg_count = zip_strategy['opportunity_count'].mean()

    # Create hover text
    zip_strategy['hover_text'] = zip_strategy.apply(
        lambda row: (
            f"<b>Zipcode {int(row['zipcode'])}</b><br>"
            f"Opportunities: {int(row['opportunity_count'])}<br>"
            f"Avg Profit: ${row['avg_potential_profit']:,.0f}<br>"
            f"Nbhd Avg Price: ${row['zip_avg_price']:,.0f}<br>"
            f"Smart Score: {row['avg_smart_score']:.1f}"
        ),
        axis=1
    )

    # Create figure
    fig = go.Figure()

    # ✅ FIX: Reduced bubble sizes (from /5000 to /8000)
    fig.add_trace(go.Scatter(
        x=zip_strategy['opportunity_count'],
        y=zip_strategy['avg_potential_profit'],
        mode='markers',
        marker=dict(
            size=zip_strategy['zip_avg_price'] / 8000,  # ✅ Smaller bubbles
            sizemin=3,  # ✅ Minimum size
            color=zip_strategy['avg_potential_profit'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="Profit Potential ($)",  # ✅ Better title
                    side="right"
                ),
                tickformat="$,.0f",
                x=1.15,  # ✅ Position colorbar more to the right
                len=0.7
            ),
            line=dict(width=1, color='black')
        ),
        text=zip_strategy['hover_text'],
        hovertemplate='%{text}<extra></extra>',
        name='Zipcodes'
    ))

    # Add reference lines
    fig.add_hline(
        y=avg_profit,
        line_dash="dash",
        line_color="red",
        opacity=0.6,
        annotation_text=f"Avg Profit: ${avg_profit / 1000:.0f}K",
        annotation_position="right"
    )

    fig.add_vline(
        x=avg_count,
        line_dash="dash",
        line_color="blue",
        opacity=0.6,
        annotation_text=f"Avg Vol: {avg_count:.0f}",
        annotation_position="top"
    )

    # Add annotations for top 5 zipcodes
    top_5 = zip_strategy.nlargest(5, 'avg_potential_profit')
    for idx, row in top_5.iterrows():
        fig.add_annotation(
            x=row['opportunity_count'],
            y=row['avg_potential_profit'],
            text=f"{int(row['zipcode'])}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="gray",
            ax=20,
            ay=-20,
            bgcolor="yellow",
            opacity=0.8,
            font=dict(size=10, color="black")
        )

    # ✅ ADD: Size legend as annotations (simulating matplotlib legend)
    # Calculate size examples
    size_examples = [240000, 320000, 400000, 480000, 560000, 640000]
    legend_y_start = zip_strategy['avg_potential_profit'].max() * 0.95
    legend_x = zip_strategy['opportunity_count'].max() * 1.05

    # Add size legend title
    fig.add_annotation(
        x=legend_x,
        y=legend_y_start,
        text="<b>Nbhd Avg Price (Size)</b>",
        showarrow=False,
        xanchor="left",
        font=dict(size=11),
        xref="x",
        yref="y"
    )

    # Add size examples
    y_offset = legend_y_start * 0.92
    for i, price in enumerate(size_examples):
        size_px = price / 8000  # Same scaling as main plot

        # Add a dummy scatter point for size reference
        fig.add_trace(go.Scatter(
            x=[legend_x],
            y=[y_offset - (i * legend_y_start * 0.08)],
            mode='markers',
            marker=dict(
                size=size_px,
                color='gray',
                line=dict(width=1, color='black')
            ),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Add text label
        fig.add_annotation(
            x=legend_x + 5,
            y=y_offset - (i * legend_y_start * 0.08),
            text=f"${price / 1000:.0f}K",
            showarrow=False,
            xanchor="left",
            font=dict(size=9),
            xref="x",
            yref="y"
        )

    # Update layout
    fig.update_layout(
        title={
            'text': 'STRATEGIC ZONES: Profit vs. Volume<br><sub>Interactive: Hover for details | Drag to zoom | Double-click to reset</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'family': 'Arial Black'}
        },
        xaxis_title="Volume of Opportunities (Number of Houses)",
        yaxis_title="Avg. Potential Profit per House",
        yaxis=dict(tickformat="$,.0f"),
        hovermode='closest',
        template='plotly_white',
        height=650,  # ✅ Slightly taller for legend
        showlegend=False,
        font=dict(size=12),
        margin=dict(r=180)  # ✅ More right margin for legends
    )

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig

