"""
Visualization functionality for the CPI Analysis & Prediction Dashboard.
Includes functions for creating charts and plots with color-blind friendly options.

This module focuses on creating data visualizations that follow best practices for
accessibility and data visualization literacy, including:

1. Using color-blind friendly palettes
2. Adding clear annotations and labels
3. Providing context through reference lines and annotations
4. Using consistent visual language
5. Implementing proper hover templates for interactive elements
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Add these imports for robust trend lines
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define color-blind friendly palettes
# Using a blue-orange palette which is more distinguishable for most color vision deficiencies
COLORBLIND_PALETTE = {
    'qualitative': ['#3288bd', '#d53e4f', '#66c2a5', '#fee08b', '#e6f598', '#abdda4'],
    'sequential': ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594'],
    'diverging': ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4']
}

# Colors for Won vs Lost (blue-orange contrast)
WON_COLOR = '#3288bd'  # Blue
LOST_COLOR = '#f58518'  # Orange

# Color scales for heatmaps (color-blind friendly)
HEATMAP_COLORSCALE_WON = 'Viridis'  # Good color-blind friendly option for sequential data
HEATMAP_COLORSCALE_LOST = 'Plasma'  # Another good color-blind friendly option

def create_type_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a pie chart showing the distribution of Won vs Lost bids.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'Type' column
    
    Returns:
        go.Figure: Plotly figure object
    """
    try:
        fig = px.pie(
            df, 
            names='Type', 
            title='Distribution of Won vs Lost Bids',
            color='Type',
            color_discrete_map={'Won': WON_COLOR, 'Lost': LOST_COLOR},
            hole=0.4
        )
        
        # Add data labels
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}'
        )
        
        # Add a more descriptive hover tooltip
        fig.update_layout(
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )
        
        # Add annotations for accessibility
        fig.update_layout(
            annotations=[
                dict(
                    text=f"Won: {(df['Type'] == 'Won').sum()} bids",
                    x=0.5,
                    y=1.1,
                    font_size=12,
                    showarrow=False
                ),
                dict(
                    text=f"Lost: {(df['Type'] == 'Lost').sum()} bids",
                    x=0.5,
                    y=1.05,
                    font_size=12,
                    showarrow=False
                )
            ]
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error in create_type_distribution_chart: {e}", exc_info=True)
        # Return empty figure
        return go.Figure()

def create_cpi_distribution_boxplot(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> go.Figure:
    """
    Create a boxplot comparing CPI distribution between Won and Lost bids.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
    
    Returns:
        go.Figure: Plotly figure object
    """
    try:
        fig = go.Figure()
        
        # Add Won trace
        fig.add_trace(go.Box(
            y=won_data['CPI'],
            name='Won',
            marker_color=WON_COLOR,
            boxmean=True,  # Show mean as a dashed line
            line=dict(width=2),
            jitter=0.3,  # Add some jitter to points for better visualization
            pointpos=-1.8,  # Offset points to the left
            boxpoints='outliers'  # Only show outliers
        ))
        
        # Add Lost trace
        fig.add_trace(go.Box(
            y=lost_data['CPI'],
            name='Lost',
            marker_color=LOST_COLOR,
            boxmean=True,  # Show mean as a dashed line
            line=dict(width=2),
            jitter=0.3,  # Add some jitter to points for better visualization
            pointpos=-1.8,  # Offset points to the left
            boxpoints='outliers'  # Only show outliers
        ))
        
        # Add mean annotations for better accessibility
        fig.add_annotation(
            x=0,  # x-position (Won)
            y=won_data['CPI'].mean(),
            text=f"Mean: ${won_data['CPI'].mean():.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            ax=50,
            ay=-30
        )
        
        fig.add_annotation(
            x=1,  # x-position (Lost)
            y=lost_data['CPI'].mean(),
            text=f"Mean: ${lost_data['CPI'].mean():.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            ax=50,
            ay=-30
        )
        
        # Update layout
        fig.update_layout(
            title='CPI Distribution: Won vs Lost',
            yaxis_title='CPI ($)',
            xaxis_title='Bid Type',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            # Add grid for easier reading of values
            yaxis=dict(
                gridcolor='rgba(0,0,0,0.1)',
                gridwidth=1
            ),
            # Add hover information template
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            ),
            # Improved accessibility with contrasting colors
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,1)',
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="black"
            )
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error in create_cpi_distribution_boxplot: {e}", exc_info=True)
        # Return empty figure
        return go.Figure()

def create_cpi_histogram_comparison(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> go.Figure:
    """
    Create a side-by-side histogram comparison of CPI distributions.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
    
    Returns:
        go.Figure: Plotly figure object
    """
    try:
        fig = make_subplots(
            rows=1, 
            cols=2, 
            subplot_titles=("Won Bids CPI Distribution", "Lost Bids CPI Distribution"),
            shared_yaxes=True,
            shared_xaxes=True
        )
        
        # Add histograms
        fig.add_trace(
            go.Histogram(
                x=won_data['CPI'], 
                name="Won", 
                marker_color=WON_COLOR, 
                opacity=0.7,
                histnorm='percent',  # Show as percentage for easier comparison
                hovertemplate='CPI: $%{x:.2f}<br>Percentage: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=lost_data['CPI'], 
                name="Lost", 
                marker_color=LOST_COLOR, 
                opacity=0.7,
                histnorm='percent',  # Show as percentage for easier comparison
                hovertemplate='CPI: $%{x:.2f}<br>Percentage: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Add vertical lines for means
        fig.add_shape(
            type="line",
            x0=won_data['CPI'].mean(), x1=won_data['CPI'].mean(),
            y0=0, y1=1,
            yref="paper",
            line=dict(color="black", width=2, dash="dash"),
            row=1, col=1
        )
        
        fig.add_shape(
            type="line",
            x0=lost_data['CPI'].mean(), x1=lost_data['CPI'].mean(),
            y0=0, y1=1,
            yref="paper",
            line=dict(color="black", width=2, dash="dash"),
            row=1, col=2
        )
        
        # Add annotations for mean values
        fig.add_annotation(
            x=won_data['CPI'].mean(),
            y=0.95,
            text=f"Mean: ${won_data['CPI'].mean():.2f}",
            showarrow=True,
            arrowhead=2,
            yref="paper",
            row=1, col=1
        )
        
        fig.add_annotation(
            x=lost_data['CPI'].mean(),
            y=0.95,
            text=f"Mean: ${lost_data['CPI'].mean():.2f}",
            showarrow=True,
            arrowhead=2,
            yref="paper",
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=500,
            showlegend=False,
            title_text="CPI Distribution Comparison (Won vs Lost)",
            # Improved accessibility with contrasting colors
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,1)',
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="black"
            )
        )
        
        fig.update_xaxes(title_text="CPI ($)")
        fig.update_yaxes(title_text="Percentage of Bids (%)")
        
        return fig
    
    except Exception as e:
        logger.error(f"Error in create_cpi_histogram_comparison: {e}", exc_info=True)
        # Return empty figure
        return go.Figure()

def create_cpi_vs_ir_scatter(won_data: pd.DataFrame, lost_data: pd.DataFrame, add_trend_line: bool = True) -> go.Figure:
    """
    Create a scatter plot of CPI vs IR with trend lines.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
        add_trend_line (bool, optional): Whether to add trend lines. Defaults to True.
    
    Returns:
        go.Figure: Plotly figure object
    """
    try:
        fig = go.Figure()
        
        # Add Won data
        fig.add_trace(go.Scatter(
            x=won_data['IR'], 
            y=won_data['CPI'], 
            mode='markers',
            marker=dict(
                color=WON_COLOR, 
                size=8, 
                opacity=0.6,
                line=dict(width=1, color='black')  # Add border for better visibility
            ),
            name="Won",
            hovertemplate='<b>Won Bid</b><br>IR: %{x:.1f}%<br>CPI: $%{y:.2f}<br>LOI: %{customdata[0]:.1f} min<br>Completes: %{customdata[1]}<extra></extra>',
            customdata=won_data[['LOI', 'Completes']]  # Include additional data for hover
        ))
        
        # Add Lost data
        fig.add_trace(go.Scatter(
            x=lost_data['IR'], 
            y=lost_data['CPI'], 
            mode='markers',
            marker=dict(
                color=LOST_COLOR, 
                size=8, 
                opacity=0.6,
                line=dict(width=1, color='black')  # Add border for better visibility
            ),
            name="Lost",
            hovertemplate='<b>Lost Bid</b><br>IR: %{x:.1f}%<br>CPI: $%{y:.2f}<br>LOI: %{customdata[0]:.1f} min<br>Completes: %{customdata[1]}<extra></extra>',
            customdata=lost_data[['LOI', 'Completes']]  # Include additional data for hover
        ))
        
        # Add trend lines
        if add_trend_line:
            # Add a trend line for Won bids
            won_trend = go.Scatter(
                x=won_data['IR'],
                y=won_data['IR'].map(lambda x: 
                    np.polyval(np.polyfit(won_data['IR'], won_data['CPI'], 1), x)
                ),
                mode='lines',
                line=dict(color=WON_COLOR, width=2, dash='solid'),
                name='Won Trend',
                hoverinfo='skip'
            )
            fig.add_trace(won_trend)
            
            # Add a trend line for Lost bids
            lost_trend = go.Scatter(
                x=lost_data['IR'],
                y=lost_data['IR'].map(lambda x: 
                    np.polyval(np.polyfit(lost_data['IR'], lost_data['CPI'], 1), x)
                ),
                mode='lines',
                line=dict(color=LOST_COLOR, width=2, dash='solid'),
                name='Lost Trend',
                hoverinfo='skip'
            )
            fig.add_trace(lost_trend)
        
        # Update layout with improved accessibility
        fig.update_layout(
            title_text="CPI vs Incidence Rate (IR) Relationship",
            height=600,
            xaxis_title="Incidence Rate (%)",
            yaxis_title="CPI ($)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            # Add grid for easier reading of values
            xaxis=dict(
                gridcolor='rgba(0,0,0,0.1)',
                gridwidth=1,
                tickformat='.0f'  # No decimal places for IR
            ),
            yaxis=dict(
                gridcolor='rgba(0,0,0,0.1)',
                gridwidth=1,
                tickprefix='$',  # Add dollar sign to y-axis
            ),
            # Improved accessibility with contrasting colors
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,1)',
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="black"
            ),
            # Add hover information template
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )
        
        # Add annotations for context
        fig.add_annotation(
            x=won_data['IR'].min() + (won_data['IR'].max() - won_data['IR'].min()) * 0.05,
            y=won_data['CPI'].max() - (won_data['CPI'].max() - won_data['CPI'].min()) * 0.05,
            text="Lower IR typically requires<br>higher CPI due to<br>increased difficulty finding<br>qualified respondents",
            showarrow=False,
            align="left",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error in create_cpi_vs_ir_scatter: {e}", exc_info=True)
        # Return empty figure
        return go.Figure()

def create_bar_chart_by_bin(won_data: pd.DataFrame, lost_data: pd.DataFrame, bin_column: str, 
                          value_column: str = 'CPI', title: str = None) -> go.Figure:
    """
    Create a bar chart comparing a value across bins between Won and Lost bids.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
        bin_column (str): Column name containing bin categories
        value_column (str, optional): Column to aggregate. Defaults to 'CPI'.
        title (str, optional): Chart title. Defaults to auto-generated title.
    
    Returns:
        go.Figure: Plotly figure object
    """
    try:
        # Generate aggregated data
        won_agg = won_data.groupby(bin_column)[value_column].mean().reset_index()
        lost_agg = lost_data.groupby(bin_column)[value_column].mean().reset_index()
        
        # Create figure
        fig = go.Figure()
        
        # Add Won bars
        fig.add_trace(go.Bar(
            x=won_agg[bin_column],
            y=won_agg[value_column],
            name='Won',
            marker_color=WON_COLOR,
            opacity=0.8,
            text=won_agg[value_column].map('${:.2f}'.format),
            textposition='auto',
            hovertemplate='<b>Won Bids</b><br>%{x}<br>Avg ' + value_column + ': $%{y:.2f}<extra></extra>'
        ))
        
        # Add Lost bars
        fig.add_trace(go.Bar(
            x=lost_agg[bin_column],
            y=lost_agg[value_column],
            name='Lost',
            marker_color=LOST_COLOR,
            opacity=0.8,
            text=lost_agg[value_column].map('${:.2f}'.format),
            textposition='auto',
            hovertemplate='<b>Lost Bids</b><br>%{x}<br>Avg ' + value_column + ': $%{y:.2f}<extra></extra>'
        ))
        
        # Generate automatic title if not provided
        if title is None:
            title = f'Average {value_column} by {bin_column}'
        
        # Determine x-axis title based on bin_column
        if bin_column == 'IR_Bin':
            xaxis_title = 'Incidence Rate Bin (%)'
        elif bin_column == 'LOI_Bin':
            xaxis_title = 'Length of Interview Bin'
        elif bin_column == 'Completes_Bin':
            xaxis_title = 'Sample Size Bin'
        else:
            xaxis_title = bin_column
        
        # Update layout with improved accessibility
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=f'Average {value_column} ($)' if value_column == 'CPI' else f'Average {value_column}',
            barmode='group',
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            # Add grid for easier reading of values
            yaxis=dict(
                gridcolor='rgba(0,0,0,0.1)',
                gridwidth=1,
                tickprefix='$' if value_column == 'CPI' else '',  # Add dollar sign if CPI
            ),
            # Improved accessibility with contrasting colors
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,1)',
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="black"
            )
        )
        
        # Add a percentage difference annotation
        for i in range(len(won_agg)):
            bin_name = won_agg[bin_column].iloc[i]
            won_val = won_agg[value_column].iloc[i]
            
            # Find matching lost value
            try:
                lost_val = lost_agg[lost_agg[bin_column] == bin_name][value_column].iloc[0]
                percent_diff = ((lost_val - won_val) / won_val) * 100
                
                if abs(percent_diff) > 10:  # Only annotate significant differences
                    fig.add_annotation(
                        x=bin_name,
                        y=max(won_val, lost_val) + 1,
                        text=f"{percent_diff:+.1f}%",
                        showarrow=False,
                        font=dict(
                            size=10,
                            color="black"
                        )
                    )
            except:
                pass
        
        return fig
    
    except Exception as e:
        logger.error(f"Error in create_bar_chart_by_bin: {e}", exc_info=True)
        # Return empty figure
        return go.Figure()

def create_heatmap(pivot_data: pd.DataFrame, title: str, colorscale: str = None) -> go.Figure:
    """
    Create a heatmap from pivot table data.
    
    Args:
        pivot_data (pd.DataFrame): Pivot table DataFrame
        title (str): Chart title
        colorscale (str, optional): Colorscale to use. Defaults to HEATMAP_COLORSCALE_WON.
    
    Returns:
        go.Figure: Plotly figure object
    """
    try:
        # Check if pivot data is empty
        if pivot_data.empty:
            logger.warning("Empty pivot data provided for heatmap.")

            # Return an empty figure with a message
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for heatmap visualization.",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16, color="black")
            )
            fig.update_layout(title=title, height=600)
            return fig
        
        # Check for all zero values
        if (pivot_data == 0).all().all():
            logger.warning("All values in pivot data are zero for heatmap.")

            # Return an empty figure with a message
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for heatmap visualization.",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16, color="black")
            )
            fig.update_layout(title=title, height=600)
            return fig

        try:
            # Use default colorscale if not specified
            if colorscale is None:
                colorscale = HEATMAP_COLORSCALE_WON
                
            # Create heatmap
            fig = px.imshow(
                pivot_data,
                labels=dict(x="LOI Bin", y="IR Bin", color="Avg CPI ($)"),
                x=pivot_data.columns,
                y=pivot_data.index,
                title=title,
                color_continuous_scale=colorscale,
                aspect="auto",
                text_auto='.2f'  # Show values on cells
            )
            
            # Update layout
            fig.update_layout(
                height=600,
                coloraxis_colorbar=dict(
                    title="Avg CPI ($)",
                    tickprefix="$",
                    len=0.75
                ),
                # Improved accessibility
                plot_bgcolor='rgba(255,255,255,1)',
                paper_bgcolor='rgba(255,255,255,1)',
                font=dict(
                    family="Arial, sans-serif",
                    size=12,
                    color="black"
                )
            )
            
            # Update xaxis properties to handle long text
            fig.update_xaxes(
                tickangle=45,
                title_font=dict(size=14),
                title_standoff=25
            )
            
            # Update yaxis properties
            fig.update_yaxes(
                title_font=dict(size=14),
                title_standoff=25
            )
            
            # Add a hover template for better interaction
            fig.update_traces(
                hovertemplate="IR Bin: %{y}<br>LOI Bin: %{x}<br>Avg CPI: $%{z:.2f}<extra></extra>"
            )
            
            return fig
        
        except Exception as e:
            logger.error(f"Error creating heatmap figure: {e}", exc_info=True)
            # Return empty figure
            return go.Figure()
    
    except Exception as e:
        logger.error(f"Error in create_heatmap: {e}", exc_info=True)
        # Return a simple empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating heatmap: {str(e)}",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(color="red", size=14)
        )
        fig.update_layout(title=title)
        return fig

def create_feature_importance_chart(feature_importance: pd.DataFrame) -> go.Figure:
    """
    Create a horizontal bar chart of feature importance.
    
    Args:
        feature_importance (pd.DataFrame): DataFrame with Feature and Importance columns
    
    Returns:
        go.Figure: Plotly figure object
    """
    try:
        # Create horizontal bar chart
        fig = px.bar(
            feature_importance, 
            x='Importance', 
            y='Feature', 
            orientation='h',
            title='Feature Importance Analysis',
            color='Importance',
            color_continuous_scale='Viridis',  # Color-blind friendly
            text=feature_importance['Importance'].map(lambda x: f"{x:.4f}")
        )
        
        # Update layout
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Relative Importance",
            yaxis_title="Feature",
            height=500,
            # Improved accessibility
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,1)',
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="black"
            )
        )
        
        # Add hover template
        fig.update_traces(
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
            textposition="outside"
        )
        
        # Add annotation explaining feature importance
        fig.add_annotation(
            x=feature_importance['Importance'].max() * 0.95,
            y=0,
            text="Higher values indicate<br>stronger influence on CPI",
            showarrow=False,
            align="right",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10)
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error in create_feature_importance_chart: {e}", exc_info=True)
        # Return empty figure
        return go.Figure()

def create_prediction_comparison_chart(predictions: dict, won_avg: float, lost_avg: float) -> go.Figure:
    """
    Create a chart comparing model predictions with won/lost averages.
    
    Args:
        predictions (dict): Dictionary of model predictions
        won_avg (float): Average CPI for won bids
        lost_avg (float): Average CPI for lost bids
    
    Returns:
        go.Figure: Plotly figure object
    """
    try:
        # Prepare data
        models = list(predictions.keys())
        values = list(predictions.values())
        
        # Add average prediction
        avg_prediction = sum(values) / len(values)
        models.append('Average Prediction')
        values.append(avg_prediction)
        
        # Add reference values
        reference_models = ['Won Avg', 'Lost Avg']
        reference_values = [won_avg, lost_avg]
        
        # Create figure
        fig = go.Figure()
        
        # Add prediction bars
        fig.add_trace(go.Bar(
            x=models,
            y=values,
            marker_color='#4292c6',  # Blue
            name='Predictions',
            text=[f"${v:.2f}" for v in values],
            textposition='auto',
            hovertemplate="<b>%{x}</b><br>CPI: $%{y:.2f}<extra></extra>"
        ))
        
        # Add reference lines
        fig.add_trace(go.Scatter(
            x=[models[0], models[-1]],
            y=[won_avg, won_avg],
            mode='lines',
            line=dict(color=WON_COLOR, width=2, dash='dot'),
            name='Won Avg',
            hovertemplate=f"Won Avg: ${won_avg:.2f}<extra></extra>"
        ))
        
        fig.add_trace(go.Scatter(
            x=[models[0], models[-1]],
            y=[lost_avg, lost_avg],
            mode='lines',
            line=dict(color=LOST_COLOR, width=2, dash='dot'),
            name='Lost Avg',
            hovertemplate=f"Lost Avg: ${lost_avg:.2f}<extra></extra>"
        ))
        
        # Update layout
        fig.update_layout(
            title='CPI Predictions Comparison',
            xaxis_title='Model',
            yaxis_title='Predicted CPI ($)',
            height=500,
            # Improved accessibility
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,1)',
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="black"
            ),
            yaxis=dict(
                gridcolor='rgba(0,0,0,0.1)',
                gridwidth=1,
                tickprefix='', # Add dollar sign to y-axis
            )
        )
        
        # Add annotations for won/lost avg
        fig.add_annotation(
            x=models[-1],
            y=won_avg,
            text=f"Won Avg: ${won_avg:.2f}",
            showarrow=True,
            arrowhead=2,
            ax=40,
            ay=20,
            font=dict(color=WON_COLOR)
        )
        
        fig.add_annotation(
            x=models[-1],
            y=lost_avg,
            text=f"Lost Avg: ${lost_avg:.2f}",
            showarrow=True,
            arrowhead=2,
            ax=40,
            ay=-20,
            font=dict(color=LOST_COLOR)
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error in create_prediction_comparison_chart: {e}", exc_info=True)
        # Return empty figure
        return go.Figure()

def create_cpi_efficiency_chart(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> go.Figure:
    """
    Create a new visualization showing CPI efficiency (IR/LOI/Completes).
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
    
    Returns:
        go.Figure: Plotly figure object
    """
    try:
        # Calculate CPI efficiency metric if not already present
        if 'CPI_Efficiency' not in won_data.columns:
            won_data['CPI_Efficiency'] = (won_data['IR'] / 100) * (1 / won_data['LOI']) * won_data['Completes']
        
        if 'CPI_Efficiency' not in lost_data.columns:
            lost_data['CPI_Efficiency'] = (lost_data['IR'] / 100) * (1 / lost_data['LOI']) * lost_data['Completes']
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter plot for Won data
        fig.add_trace(go.Scatter(
            x=won_data['CPI_Efficiency'],
            y=won_data['CPI'],
            mode='markers',
            marker=dict(
                color=WON_COLOR,
                size=10,
                opacity=0.6,
                line=dict(width=1, color='black')
            ),
            name='Won',
            hovertemplate='<b>Won Bid</b><br>Efficiency: %{x:.1f}<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<br>LOI: %{customdata[1]:.1f} min<br>Completes: %{customdata[2]}<extra></extra>',
            customdata=won_data[['IR', 'LOI', 'Completes']]
        ))
        
        # Add scatter plot for Lost data
        fig.add_trace(go.Scatter(
            x=lost_data['CPI_Efficiency'],
            y=lost_data['CPI'],
            mode='markers',
            marker=dict(
                color=LOST_COLOR,
                size=10,
                opacity=0.6,
                line=dict(width=1, color='black')
            ),
            name='Lost',
            hovertemplate='<b>Lost Bid</b><br>Efficiency: %{x:.1f}<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<br>LOI: %{customdata[1]:.1f} min<br>Completes: %{customdata[2]}<extra></extra>',
            customdata=lost_data[['IR', 'LOI', 'Completes']]
        ))
        
        # Add trend lines
        # Won trend line
        x_range = np.linspace(won_data['CPI_Efficiency'].min(), won_data['CPI_Efficiency'].max(), 100)
        coeffs = np.polyfit(won_data['CPI_Efficiency'], won_data['CPI'], 1)
        trend_y = np.polyval(coeffs, x_range)
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=trend_y,
            mode='lines',
            line=dict(color=WON_COLOR, width=2),
            name='Won Trend',
            hoverinfo='skip'
        ))
        
        # Lost trend line
        x_range = np.linspace(lost_data['CPI_Efficiency'].min(), lost_data['CPI_Efficiency'].max(), 100)
        coeffs = np.polyfit(lost_data['CPI_Efficiency'], lost_data['CPI'], 1)
        trend_y = np.polyval(coeffs, x_range)
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=trend_y,
            mode='lines',
            line=dict(color=LOST_COLOR, width=2),
            name='Lost Trend',
            hoverinfo='skip'
        ))
        
        # Update layout
        fig.update_layout(
            title='CPI vs Efficiency Metric',
            xaxis_title='Efficiency Metric ((IR/100) × (1/LOI) × Completes)',
            yaxis_title='CPI ($)',
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            # Improved accessibility
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,1)',
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="black"
            ),
            yaxis=dict(
                gridcolor='rgba(0,0,0,0.1)',
                gridwidth=1,
                tickprefix='',  # Add dollar sign to y-axis
            ),
            xaxis=dict(
                gridcolor='rgba(0,0,0,0.1)',
                gridwidth=1
            )
        )
        
        # Add annotation explaining the efficiency metric
        fig.add_annotation(
            x=0.02,
            y=0.95,
            xref="paper",
            yref="paper",
            text="Efficiency Metric combines IR, LOI, and<br>Sample Size into a single value.<br>Higher values indicate more<br>efficient survey parameters.",
            showarrow=False,
            align="left",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error in create_cpi_efficiency_chart: {e}", exc_info=True)
        # Return empty figure
        return go.Figure()

if __name__ == "__main__":
    # Test the visualization functions with sample data
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Create sample data
        won_data = pd.DataFrame({
            'CPI': [10.5, 8.7, 9.8, 11.2, 7.5],
            'IR': [25, 45, 55, 20, 60],
            'LOI': [10, 8, 12, 15, 5],
            'Completes': [500, 800, 600, 400, 900],
            'Type': ['Won'] * 5,
            'IR_Bin': ['20-30', '40-50', '50-60', '10-20', '50-60'],
            'LOI_Bin': ['Short (6-10 min)', 'Very Short (1-5 min)', 'Short (6-10 min)', 
                       'Medium (11-15 min)', 'Very Short (1-5 min)'],
            'Completes_Bin': ['Medium (101-500)', 'Small (1-100)', 'Small (1-100)', 
                             'Medium (101-500)', 'Small (1-100)']
        })
        
        lost_data = pd.DataFrame({
            'CPI': [12.3, 15.2, 14.1, 13.5, 16.8],
            'IR': [35, 15, 40, 30, 25],
            'LOI': [15, 20, 18, 12, 25],
            'Completes': [300, 200, 250, 400, 150],
            'Type': ['Lost'] * 5,
            'IR_Bin': ['30-40', '10-20', '30-40', '20-30', '20-30'],
            'LOI_Bin': ['Medium (11-15 min)', 'Long (16-20 min)', 'Long (16-20 min)', 
                       'Short (6-10 min)', 'Very Long (20+ min)'],
            'Completes_Bin': ['Medium (101-500)', 'Small (1-100)', 'Small (1-100)', 
                             'Medium (101-500)', 'Small (1-100)']
        })

        # Calculate efficiency metric
        won_data['CPI_Efficiency'] = (won_data['IR'] / 100) * (1 / won_data['LOI']) * won_data['Completes']
        lost_data['CPI_Efficiency'] = (lost_data['IR'] / 100) * (1 / lost_data['LOI']) * lost_data['Completes']
        
        # Create pivot tables for heatmap
        won_pivot = won_data.pivot_table(
            values='CPI', 
            index='IR_Bin', 
            columns='LOI_Bin', 
            aggfunc='mean'
        ).fillna(0)
        
        lost_pivot = lost_data.pivot_table(
            values='CPI', 
            index='IR_Bin', 
            columns='LOI_Bin', 
            aggfunc='mean'
        ).fillna(0)
        
        # Sample feature importance
        feature_importance = pd.DataFrame({
            'Feature': ['IR', 'LOI', 'Completes', 'IR_LOI_Ratio', 'Type_Won'],
            'Importance': [0.35, 0.25, 0.20, 0.15, 0.05]
        }).sort_values('Importance', ascending=False)
        
        # Test type distribution chart
        combined_data = pd.concat([won_data, lost_data])
        type_chart = create_type_distribution_chart(combined_data)
        print("Type distribution chart created successfully")
        
        # Test CPI distribution boxplot
        boxplot = create_cpi_distribution_boxplot(won_data, lost_data)
        print("CPI distribution boxplot created successfully")
        
        # Test CPI histogram comparison
        histogram = create_cpi_histogram_comparison(won_data, lost_data)
        print("CPI histogram comparison created successfully")
        
        # Test CPI vs IR scatter
        scatter = create_cpi_vs_ir_scatter(won_data, lost_data)
        print("CPI vs IR scatter created successfully")
        
        # Test bar chart by bin
        bar_chart = create_bar_chart_by_bin(won_data, lost_data, 'IR_Bin')
        print("Bar chart by bin created successfully")
        
        # Test heatmap
        heatmap = create_heatmap(won_pivot, "Won Deals: Average CPI by IR and LOI")
        print("Heatmap created successfully")
        
        # Test feature importance chart
        feature_chart = create_feature_importance_chart(feature_importance)
        print("Feature importance chart created successfully")
        
        # Test prediction comparison chart
        predictions = {'Linear Regression': 11.2, 'Random Forest': 10.8, 'Gradient Boosting': 11.5}
        prediction_chart = create_prediction_comparison_chart(predictions, 10.0, 14.0)
        print("Prediction comparison chart created successfully")
        
        # Test CPI efficiency chart
        efficiency_chart = create_cpi_efficiency_chart(won_data, lost_data)
        print("CPI efficiency chart created successfully")
        
        print("All visualization tests completed successfully")
        
    except Exception as e:
        print(f"Error testing visualizations: {e}")
