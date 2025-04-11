"""
CPI Analysis component for the CPI Analysis & Prediction Dashboard.
Displays detailed analysis of CPI by different factors.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Any, Optional
import traceback

# Import visualization utilities
from utils.visualization import (
    create_cpi_histogram_comparison,
    create_cpi_vs_ir_scatter,
    create_bar_chart_by_bin,
    create_heatmap
)

# Import data processing utilities
from utils.data_processor import handle_outliers

# Import constants
from utils.visualization import (
    HEATMAP_COLORSCALE_WON,
    HEATMAP_COLORSCALE_LOST
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def show_analysis(won_data: pd.DataFrame, lost_data: pd.DataFrame, combined_data: pd.DataFrame) -> None:
    """
    Display the CPI analysis dashboard showing detailed breakdown by different factors.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
        combined_data (pd.DataFrame): Combined DataFrame of won and lost bids
    """
    try:
        st.title("CPI Analysis: Won vs. Lost Bids")
        
        # Introduction
        st.markdown("""
        This section provides a detailed analysis of Cost Per Interview (CPI) by different
        factors that influence pricing. Use the tabs below to explore different analyses.
        """)
        
        # Create tabs for different analyses
        tabs = st.tabs([
            "CPI Distribution", 
            "By Incidence Rate (IR)", 
            "By Length of Interview (LOI)", 
            "By Sample Size",
            "Multi-Factor Analysis"
        ])
        
        # CPI Distribution tab
        with tabs[0]:
            st.subheader("CPI Distribution Comparison")
            
            # Add description
            st.markdown("""
            This analysis shows the distribution of CPI values for won and lost bids.
            The histograms help visualize the range and frequency of different price points
            and identify potential pricing thresholds.
            """)
            
            # Create and display CPI histogram comparison
            try:
                fig = create_cpi_histogram_comparison(won_data, lost_data)
                st.plotly_chart(fig, use_container_width=True, key='cpi_hist_comparison')
            except Exception as e:
                st.error(f"Error creating CPI histogram: {str(e)}")
                logger.error(f"Error in CPI histogram: {e}", exc_info=True)
            
            # Add CPI statistics
            col1, col2 = st.columns(2)
            
            # Calculate statistics with outlier handling
            won_clean = handle_outliers(won_data, ['CPI'], method='percentile', threshold=0.95)
            lost_clean = handle_outliers(lost_data, ['CPI'], method='percentile', threshold=0.95)
            
            with col1:
                st.subheader("Won Bids CPI Statistics")
                st.markdown(f"""
                - **Minimum:** ${won_clean['CPI'].min():.2f}
                - **Maximum:** ${won_clean['CPI'].max():.2f}
                - **Mean:** ${won_clean['CPI'].mean():.2f}
                - **Median:** ${won_clean['CPI'].median():.2f}
                - **Standard Deviation:** ${won_clean['CPI'].std():.2f}
                """)
            
            with col2:
                st.subheader("Lost Bids CPI Statistics")
                st.markdown(f"""
                - **Minimum:** ${lost_clean['CPI'].min():.2f}
                - **Maximum:** ${lost_clean['CPI'].max():.2f}
                - **Mean:** ${lost_clean['CPI'].mean():.2f}
                - **Median:** ${lost_clean['CPI'].median():.2f}
                - **Standard Deviation:** ${lost_clean['CPI'].std():.2f}
                """)
            
            # Add interpretation
            with st.expander("📊 Interpretation"):
                st.markdown("""
                ### What This Analysis Shows
                
                The histograms display the distribution of CPI values for won and lost bids, 
                revealing the price ranges that are most common in each category.
                
                ### Key Insights
                
                1. **Price Range Comparison**: Won bids typically have a lower CPI range compared to lost bids,
                   indicating that competitive pricing is important for winning projects.
                
                2. **Distribution Shape**: The shape of the distribution provides insights into pricing patterns.
                   A narrower distribution suggests more consistent pricing, while a wider distribution indicates
                   more variable pricing based on project factors.
                
                3. **Overlap Areas**: Where the distributions overlap represents the competitive pricing zone
                   where other factors besides price (such as reputation, capabilities, relationships) may
                   determine bid success.
                
                4. **Pricing Thresholds**: The mean values (vertical dashed lines) can be used as reference
                   points for setting competitive pricing thresholds.
                """)
        
        # By Incidence Rate (IR) tab
        with tabs[1]:
            st.subheader("CPI Analysis by Incidence Rate (IR)")
            
            # Add description
            st.markdown("""
            This analysis explores how Incidence Rate (IR) - the percentage of people who qualify for a survey -
            affects CPI. Lower IR usually means it's harder to find qualified respondents, potentially
            justifying higher prices.
            """)
            
            # Create and display CPI vs IR scatter plot
            try:
                fig = create_cpi_vs_ir_scatter(won_clean, lost_clean)
                st.plotly_chart(fig, use_container_width=True, key='cpi_vs_ir_scatter')
            except Exception as e:
                st.error(f"Error creating IR scatter plot: {str(e)}")
                logger.error(f"Error in IR scatter plot: {e}", exc_info=True)
            
            # Create and display CPI by IR Bin bar chart
            st.subheader("Average CPI by IR Bin")
            try:
                fig = create_bar_chart_by_bin(won_clean, lost_clean, 'IR_Bin', 'CPI',
                                        title='Average CPI by Incidence Rate Bin')
                st.plotly_chart(fig, use_container_width=True, key='cpi_by_ir_bin')
            except Exception as e:
                st.error(f"Error creating IR bin chart: {str(e)}")
                logger.error(f"Error in IR bin chart: {e}", exc_info=True)
            
            # Add interpretation
            with st.expander("📊 Interpretation"):
                st.markdown("""
                ### Understanding the IR-CPI Relationship
                
                Incidence Rate (IR) is the percentage of people who qualify for a survey. It has a significant
                impact on CPI because it affects how difficult it is to find qualified respondents.
                
                ### Key Insights
                
                1. **Inverse Relationship**: As shown in both charts, there's generally an inverse relationship
                   between IR and CPI - as IR increases, CPI tends to decrease. This is logical because higher
                   incidence rates mean less screening effort is needed.
                
                2. **Price Gap by IR Level**: The bar chart reveals that the gap between won and lost CPIs
                   varies across IR bins. This can help identify where competitive pricing sensitivity is highest.
                
                3. **Pricing Strategy**: For lower IR ranges (0-30%), pricing sensitivity appears higher,
                   suggesting that competitive pricing is especially important for low-IR projects.
                
                4. **Diminishing Returns**: The benefit of higher IR on CPI appears to flatten above certain
                   IR thresholds, suggesting that very high IR doesn't necessarily enable proportionally lower pricing.
                """)
        
        # By Length of Interview (LOI) tab
        with tabs[2]:
            st.subheader("CPI Analysis by Length of Interview (LOI)")
            
            # Add description
            st.markdown("""
            This analysis explores how Length of Interview (LOI) - the duration of the survey in minutes -
            affects CPI. Longer surveys typically command higher prices to compensate respondents for
            their additional time.
            """)
            
            # Create CPI vs LOI scatter plot with error handling
            try:
                # Create CPI vs LOI scatter plot
                fig = go.Figure()
                
                # Add Won data
                fig.add_trace(go.Scatter(
                    x=won_clean['LOI'], 
                    y=won_clean['CPI'], 
                    mode='markers',
                    marker=dict(color='#3288bd', size=8, opacity=0.6),
                    name="Won",
                    hovertemplate='<b>Won Bid</b><br>LOI: %{x:.1f} min<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<extra></extra>',
                    customdata=won_clean[['IR']]
                ))
                
                # Add Lost data
                fig.add_trace(go.Scatter(
                    x=lost_clean['LOI'], 
                    y=lost_clean['CPI'], 
                    mode='markers',
                    marker=dict(color='#f58518', size=8, opacity=0.6),
                    name="Lost",
                    hovertemplate='<b>Lost Bid</b><br>LOI: %{x:.1f} min<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<extra></extra>',
                    customdata=lost_clean[['IR']]
                ))
                
                # Try to add trend lines with robust error handling
                try:
                    # Won trend line
                    won_trend_x = np.linspace(won_clean['LOI'].min(), won_clean['LOI'].max(), 100)
                    # Use robust linear regression (ignore outliers)
                    from sklearn.linear_model import RANSACRegressor
                    
                    # Won trend line
                    won_X = won_clean['LOI'].values.reshape(-1, 1)
                    won_y = won_clean['CPI'].values
                    ransac = RANSACRegressor(random_state=42)
                    ransac.fit(won_X, won_y)
                    won_trend_y = ransac.predict(won_trend_x.reshape(-1, 1))
                    
                    fig.add_trace(go.Scatter(
                        x=won_trend_x,
                        y=won_trend_y,
                        mode='lines',
                        line=dict(color='#3288bd', width=2),
                        name='Won Trend',
                        hoverinfo='skip'
                    ))
                    
                    # Lost trend line
                    lost_trend_x = np.linspace(lost_clean['LOI'].min(), lost_clean['LOI'].max(), 100)
                    lost_X = lost_clean['LOI'].values.reshape(-1, 1)
                    lost_y = lost_clean['CPI'].values
                    ransac = RANSACRegressor(random_state=42)
                    ransac.fit(lost_X, lost_y)
                    lost_trend_y = ransac.predict(lost_trend_x.reshape(-1, 1))
                    
                    fig.add_trace(go.Scatter(
                        x=lost_trend_x,
                        y=lost_trend_y,
                        mode='lines',
                        line=dict(color='#f58518', width=2),
                        name='Lost Trend',
                        hoverinfo='skip'
                    ))
                except Exception as e:
                    logger.warning(f"Could not add trend lines to LOI scatter plot: {e}")
                    # Fall back to simple polyfit if RANSAC fails
                    try:
                        # Simple polyfit as fallback
                        won_trend_x = np.linspace(won_clean['LOI'].min(), won_clean['LOI'].max(), 100)
                        won_coeffs = np.polyfit(won_clean['LOI'], won_clean['CPI'], 1)
                        won_trend_y = np.polyval(won_coeffs, won_trend_x)
                        
                        fig.add_trace(go.Scatter(
                            x=won_trend_x,
                            y=won_trend_y,
                            mode='lines',
                            line=dict(color='#3288bd', width=2),
                            name='Won Trend',
                            hoverinfo='skip'
                        ))
                        
                        # Lost trend line
                        lost_trend_x = np.linspace(lost_clean['LOI'].min(), lost_clean['LOI'].max(), 100)
                        lost_coeffs = np.polyfit(lost_clean['LOI'], lost_clean['CPI'], 1)
                        lost_trend_y = np.polyval(lost_coeffs, lost_trend_x)
                        
                        fig.add_trace(go.Scatter(
                            x=lost_trend_x,
                            y=lost_trend_y,
                            mode='lines',
                            line=dict(color='#f58518', width=2),
                            name='Lost Trend',
                            hoverinfo='skip'
                        ))
                    except Exception as e:
                        logger.warning(f"Could not add simple trend lines either: {e}")
                        # Continue without trend lines
                
                # Update layout
                fig.update_layout(
                    title_text="CPI vs Length of Interview (LOI) Relationship",
                    height=500,
                    xaxis_title="Length of Interview (minutes)",
                    yaxis_title="CPI ($)",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    plot_bgcolor='rgba(255,255,255,1)',
                    paper_bgcolor='rgba(255,255,255,1)',
                    xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
                    yaxis=dict(gridcolor='rgba(0,0,0,0.1)', tickprefix='$')
                )
                
                st.plotly_chart(fig, use_container_width=True, key='cpi_vs_loi_scatter')
            except Exception as e:
                st.error(f"Error creating LOI scatter plot: {str(e)}")
                logger.error(f"Error in LOI scatter plot: {e}", exc_info=True)
            
            # Create and display CPI by LOI Bin bar chart
            st.subheader("Average CPI by LOI Bin")
            try:
                fig = create_bar_chart_by_bin(won_clean, lost_clean, 'LOI_Bin', 'CPI',
                                            title='Average CPI by Length of Interview Bin')
                st.plotly_chart(fig, use_container_width=True, key='cpi_by_loi_bin')
            except Exception as e:
                st.error(f"Error creating LOI bin chart: {str(e)}")
                logger.error(f"Error in LOI bin chart: {e}", exc_info=True)
            
            # Add interpretation
            with st.expander("📊 Interpretation"):
                st.markdown("""
                ### Understanding the LOI-CPI Relationship
                
                Length of Interview (LOI) is the duration of the survey in minutes. It directly affects
                respondent compensation and therefore influences the overall CPI.
                
                ### Key Insights
                
                1. **Positive Correlation**: Both charts show a clear positive correlation between LOI and CPI.
                   As surveys get longer, prices increase to compensate respondents for their time.
                
                2. **Pricing Slope**: The trend lines in the scatter plot show how CPI typically increases with
                   each additional minute of survey length. This can be used as a guideline for pricing adjustment.
                
                3. **Won vs. Lost Comparison**: The gap between won and lost bids widens as LOI increases,
                   suggesting that competitive pricing becomes even more critical for longer surveys.
                
                4. **Pricing Thresholds**: The bar chart reveals clear pricing thresholds for different LOI bins,
                   which can serve as benchmarks when pricing new projects.
                """)
        
        # By Sample Size tab
        with tabs[3]:
            st.subheader("CPI Analysis by Sample Size (Completes)")
            
            # Add description
            st.markdown("""
            This analysis explores how Sample Size (the number of completed interviews) affects CPI.
            Larger samples typically benefit from volume discounts, resulting in lower per-unit costs.
            """)
            
            # Create CPI vs Completes scatter plot with error handling
            try:
                # Create CPI vs Completes scatter plot
                fig = go.Figure()
                
                # Add Won data
                fig.add_trace(go.Scatter(
                    x=won_clean['Completes'], 
                    y=won_clean['CPI'], 
                    mode='markers',
                    marker=dict(color='#3288bd', size=8, opacity=0.6),
                    name="Won",
                    hovertemplate='<b>Won Bid</b><br>Completes: %{x}<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<br>LOI: %{customdata[1]:.1f} min<extra></extra>',
                    customdata=won_clean[['IR', 'LOI']]
                ))
                
                # Add Lost data
                fig.add_trace(go.Scatter(
                    x=lost_clean['Completes'], 
                    y=lost_clean['CPI'], 
                    mode='markers',
                    marker=dict(color='#f58518', size=8, opacity=0.6),
                    name="Lost",
                    hovertemplate='<b>Lost Bid</b><br>Completes: %{x}<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<br>LOI: %{customdata[1]:.1f} min<extra></extra>',
                    customdata=lost_clean[['IR', 'LOI']]
                ))
                
                # Try to add trend lines with robust error handling
                try:
                    # Log transform the x-axis for better visualization of relationship
                    # Add a small constant to handle zeros
                    epsilon = 1  # Small constant to add for log transform
                    
                    # Won trend
                    won_log_x = np.log(won_clean['Completes'] + epsilon)
                    won_y = won_clean['CPI']
                    
                    # Filter valid values
                    mask = ~np.isnan(won_log_x) & ~np.isnan(won_y) & ~np.isinf(won_log_x)
                    won_log_x = won_log_x[mask]
                    won_y = won_y[mask]
                    
                    if len(won_log_x) > 1:  # Need at least 2 points for regression
                        won_coeffs = np.polyfit(won_log_x, won_y, 1)
                        
                        won_trend_x = np.linspace(won_clean['Completes'].min(), won_clean['Completes'].max(), 100)
                        won_trend_log_x = np.log(won_trend_x + epsilon)
                        won_trend_y = np.polyval(won_coeffs, won_trend_log_x)
                        
                        fig.add_trace(go.Scatter(
                            x=won_trend_x,
                            y=won_trend_y,
                            mode='lines',
                            line=dict(color='#3288bd', width=2),
                            name='Won Trend',
                            hoverinfo='skip'
                        ))
                    
                    # Lost trend
                    lost_log_x = np.log(lost_clean['Completes'] + epsilon)
                    lost_y = lost_clean['CPI']
                    
                    # Filter valid values
                    mask = ~np.isnan(lost_log_x) & ~np.isnan(lost_y) & ~np.isinf(lost_log_x)
                    lost_log_x = lost_log_x[mask]
                    lost_y = lost_y[mask]
                    
                    if len(lost_log_x) > 1:  # Need at least 2 points for regression
                        lost_coeffs = np.polyfit(lost_log_x, lost_y, 1)
                        
                        lost_trend_x = np.linspace(lost_clean['Completes'].min(), lost_clean['Completes'].max(), 100)
                        lost_trend_log_x = np.log(lost_trend_x + epsilon)
                        lost_trend_y = np.polyval(lost_coeffs, lost_trend_log_x)
                        
                        fig.add_trace(go.Scatter(
                            x=lost_trend_x,
                            y=lost_trend_y,
                            mode='lines',
                            line=dict(color='#f58518', width=2),
                            name='Lost Trend',
                            hoverinfo='skip'
                        ))
                except Exception as e:
                    logger.warning(f"Could not add trend lines to Completes scatter plot: {e}")
                    # Continue without trend lines
                
                # Update layout
                fig.update_layout(
                    title_text="CPI vs Sample Size (Completes) Relationship",
                    height=500,
                    xaxis_title="Sample Size (Number of Completes)",
                    yaxis_title="CPI ($)",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    plot_bgcolor='rgba(255,255,255,1)',
                    paper_bgcolor='rgba(255,255,255,1)',
                    xaxis=dict(gridcolor='rgba(0,0,0,0.1)', type='log'),  # Log scale for better visualization
                    yaxis=dict(gridcolor='rgba(0,0,0,0.1)', tickprefix='$')
                )
                
                st.plotly_chart(fig, use_container_width=True, key='cpi_vs_completes_scatter')
            except Exception as e:
                st.error(f"Error creating Completes scatter plot: {str(e)}")
                logger.error(f"Error in Completes scatter plot: {e}", exc_info=True)
            
            # Create and display CPI by Completes Bin bar chart
            st.subheader("Average CPI by Sample Size Bin")
            try:
                fig = create_bar_chart_by_bin(won_clean, lost_clean, 'Completes_Bin', 'CPI',
                                            title='Average CPI by Sample Size Bin')
                st.plotly_chart(fig, use_container_width=True, key='cpi_by_completes_bin')
            except Exception as e:
                st.error(f"Error creating Completes bin chart: {str(e)}")
                logger.error(f"Error in Completes bin chart: {e}", exc_info=True)
            
            # Add interpretation
            with st.expander("📊 Interpretation"):
                st.markdown("""
                ### Understanding the Sample Size-CPI Relationship
                
                Sample size (number of completes) affects CPI through economies of scale - larger projects
                typically have lower per-unit costs due to volume efficiencies.
                
                ### Key Insights
                
                1. **Negative Correlation**: Both charts show that as sample size increases, CPI tends to decrease,
                   following a logarithmic curve (diminishing returns). This reflects standard volume discount practices.
                
                2. **Volume Discount Thresholds**: The bar chart reveals clear pricing thresholds at different
                   sample size bins, providing guidance on appropriate volume discount levels.
                
                3. **Won vs. Lost Comparison**: The gap between won and lost bids changes across sample size bins,
                   suggesting different pricing sensitivities at different volumes.
                
                4. **Large Sample Competitiveness**: The competitive gap appears larger for very large samples,
                   indicating that pricing competitiveness may be especially important for high-volume projects.
                """)
        
        # Multi-Factor Analysis tab
        with tabs[4]:
            st.subheader("Multi-Factor Analysis")
            
            # Add description
            st.markdown("""
            This analysis examines how CPI is influenced by multiple factors simultaneously,
            particularly focusing on the combined effect of Incidence Rate (IR) and Length of Interview (LOI).
            """)
            
            # Create pivot tables for heatmaps with robust error handling
            try:
                # Use the cleaned data for the heatmaps
                # Handle potential NaN or empty values in bins
                won_clean = won_clean.dropna(subset=['IR_Bin', 'LOI_Bin', 'CPI'])
                lost_clean = lost_clean.dropna(subset=['IR_Bin', 'LOI_Bin', 'CPI'])
                
                # Create pivot tables
                won_pivot = won_clean.pivot_table(
                    values='CPI', 
                    index='IR_Bin', 
                    columns='LOI_Bin', 
                    aggfunc='mean'
                )
                
                lost_pivot = lost_clean.pivot_table(
                    values='CPI', 
                    index='IR_Bin', 
                    columns='LOI_Bin', 
                    aggfunc='mean'
                )
                
                # Check if pivot tables are empty
                if won_pivot.empty or lost_pivot.empty:
                    st.warning("Not enough data to create heatmaps. Try adjusting your filters.")
                else:
                    # Display heatmaps
                    st.subheader("IR and LOI Combined Influence on CPI")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Won deals heatmap
                        try:
                            fig = create_heatmap(
                                won_pivot, 
                                "Won Deals: Average CPI by IR and LOI",
                                HEATMAP_COLORSCALE_WON
                            )
                            st.plotly_chart(fig, use_container_width=True, key='heatmap_won')
                        except Exception as e:
                            st.error(f"Error creating Won deals heatmap: {str(e)}")
                            logger.error(f"Error in Won deals heatmap: {e}", exc_info=True)
                    
                    with col2:
                        # Lost deals heatmap
                        try:
                            fig = create_heatmap(
                                lost_pivot, 
                                "Lost Deals: Average CPI by IR and LOI",
                                HEATMAP_COLORSCALE_LOST
                            )
                            st.plotly_chart(fig, use_container_width=True, key='heatmap_lost')
                        except Exception as e:
                            st.error(f"Error creating Lost deals heatmap: {str(e)}")
                            logger.error(f"Error in Lost deals heatmap: {e}", exc_info=True)
                    
                    # Create a differential heatmap
                    st.subheader("CPI Differential: Lost vs. Won")
                    
                    try:
                        # Fill NaN values with column medians to handle sparse data
                        for pivot in [won_pivot, lost_pivot]:
                            for col in pivot.columns:
                                pivot[col] = pivot[col].fillna(pivot[col].median())
                        
                        # Reindex both pivots to have the same shape
                        all_indices = won_pivot.index.union(lost_pivot.index)
                        all_columns = won_pivot.columns.union(lost_pivot.columns)
                        
                        won_pivot_filled = won_pivot.reindex(index=all_indices, columns=all_columns).fillna(won_pivot.median().median())
                        lost_pivot_filled = lost_pivot.reindex(index=all_indices, columns=all_columns).fillna(lost_pivot.median().median())
                        
                        # Calculate CPI differential
                        diff_pivot = lost_pivot_filled - won_pivot_filled
                        
                        # Create heatmap for differential
                        fig = px.imshow(
                            diff_pivot,
                            labels=dict(x="LOI Bin", y="IR Bin", color="CPI Difference ($)"),
                            x=diff_pivot.columns,
                            y=diff_pivot.index,
                            title="CPI Differential: Lost Minus Won",
                            color_continuous_scale="RdBu_r",  # Red-Blue diverging colorscale (red for positive, blue for negative)
                            aspect="auto",
                            text_auto='.2f'  # Show values on cells
                        )
                        
                        # Update layout
                        fig.update_layout(
                            height=600,
                            coloraxis_colorbar=dict(
                                title="CPI Diff ($)",
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
                        
                        st.plotly_chart(fig, use_container_width=True, key='heatmap_diff')
                    except Exception as e:
                        st.error(f"Error creating differential heatmap: {str(e)}")
                        logger.error(f"Error in differential heatmap: {e}", exc_info=True)
                        # Try a simplified approach if the detailed one fails
                        try:
                            # Create a simpler differential heatmap without as much preprocessing
                            diff_pivot_simple = lost_pivot.fillna(0) - won_pivot.fillna(0)
                            
                            fig = px.imshow(
                                diff_pivot_simple,
                                labels=dict(x="LOI Bin", y="IR Bin", color="CPI Difference ($)"),
                                title="CPI Differential: Lost Minus Won (Simplified)",
                                color_continuous_scale="RdBu_r",
                                aspect="auto"
                            )
                            
                            fig.update_layout(height=600)
                            st.plotly_chart(fig, use_container_width=True, key='heatmap_diff_simple')
                        except Exception as e2:
                            st.error("Could not create differential heatmap with either method. Please check your data.")
            
            # Add interpretation
            with st.expander("📊 Interpretation"):
                st.markdown("""
                ### Understanding Multi-Factor Effects on CPI
                
                The heatmaps visualize how IR and LOI jointly influence CPI, revealing complex patterns
                that might not be evident when examining each factor in isolation.
                
                ### Key Insights
                
                1. **Interaction Effect**: The heatmaps show that IR and LOI interact to influence CPI.
                   For example, the impact of low IR is more pronounced for longer surveys, creating
                   a compound effect on pricing.
                
                2. **Pricing Hot Spots**: Dark areas in the won/lost heatmaps indicate combinations of IR and LOI
                   that command higher prices, helping identify premium pricing opportunities.
                
                3. **Competitive Gaps**: The differential heatmap reveals where the largest pricing gaps exist
                   between won and lost bids, highlighting where pricing adjustments might have the biggest impact.
                
                4. **Optimal Pricing Zones**: Areas with smaller differentials represent zones where won and lost
                   bids have similar pricing, suggesting highly competitive price points.
                """)
                
            # Additional multi-factor analysis - implement with robust error handling
            st.subheader("Combined Factor Impact on Total Project Cost")
            
            try:
                # Calculate total project cost
                won_clean['Total_Cost'] = won_clean['CPI'] * won_clean['Completes']
                lost_clean['Total_Cost'] = lost_clean['CPI'] * lost_clean['Completes']
                
                # Cap total cost to prevent visualization issues
                total_cost_cap = np.percentile(
                    np.concatenate([won_clean['Total_Cost'].dropna(), lost_clean['Total_Cost'].dropna()]), 
                    95
                )
                won_clean.loc[won_clean['Total_Cost'] > total_cost_cap, 'Total_Cost'] = total_cost_cap
                lost_clean.loc[lost_clean['Total_Cost'] > total_cost_cap, 'Total_Cost'] = total_cost_cap
                
                # Create 3D scatter plot
                fig = go.Figure()
                
                # Add Won data
                fig.add_trace(go.Scatter3d(
                    x=won_clean['IR'],
                    y=won_clean['LOI'],
                    z=won_clean['Total_Cost'],
                    mode='markers',
                    marker=dict(
                        size=won_clean['Completes'] / max(50, won_clean['Completes'].max() / 50),  # Size based on completes
                        color='#3288bd',
                        opacity=0.7,
                        symbol='circle'
                    ),
                    name='Won',
                    hovertemplate='<b>Won Bid</b><br>IR: %{x:.1f}%<br>LOI: %{y:.1f} min<br>Total Cost: $%{z:.2f}<br>Completes: %{customdata[0]}<br>CPI: $%{customdata[1]:.2f}<extra></extra>',
                    customdata=won_clean[['Completes', 'CPI']]
                ))
                
                # Add Lost data
                fig.add_trace(go.Scatter3d(
                    x=lost_clean['IR'],
                    y=lost_clean['LOI'],
                    z=lost_clean['Total_Cost'],
                    mode='markers',
                    marker=dict(
                        size=lost_clean['Completes'] / max(50, lost_clean['Completes'].max() / 50),  # Size based on completes
                        color='#f58518',
                        opacity=0.7,
                        symbol='circle'
                    ),
                    name='Lost',
                    hovertemplate='<b>Lost Bid</b><br>IR: %{x:.1f}%<br>LOI: %{y:.1f} min<br>Total Cost: $%{z:.2f}<br>Completes: %{customdata[0]}<br>CPI: $%{customdata[1]:.2f}<extra></extra>',
                    customdata=lost_clean[['Completes', 'CPI']]
                ))
                
                # Update layout
                fig.update_layout(
                    title='3D Visualization: IR, LOI, and Total Project Cost',
                    scene=dict(
                        xaxis_title='Incidence Rate (%)',
                        yaxis_title='Length of Interview (min)',
                        zaxis_title='Total Project Cost ($)',
                        xaxis=dict(backgroundcolor='rgb(255, 255, 255)'),
                        yaxis=dict(backgroundcolor='rgb(255, 255, 255)'),
                        zaxis=dict(backgroundcolor='rgb(255, 255, 255)')
                    ),
                    height=700,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True, key='3d_scatter')
            except Exception as e:
                st.error(f"Error creating 3D visualization: {str(e)}")
                logger.error(f"Error in 3D visualization: {e}", exc_info=True)
                
                # Fall back to a 2D visualization if 3D fails
                try:
                    # Create a 2D scatter plot with bubble size for Total Cost
                    fig = px.scatter(
                        pd.concat([
                            won_clean.assign(Type='Won'),
                            lost_clean.assign(Type='Lost')
                        ]),
                        x='IR',
                        y='LOI',
                        size='Total_Cost',
                        color='Type',
                        color_discrete_map={'Won': '#3288bd', 'Lost': '#f58518'},
                        hover_name='Type',
                        hover_data=['CPI', 'Completes', 'Total_Cost'],
                        title='Alternative View: IR, LOI, and Total Project Cost',
                    )
                    
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True, key='bubble_chart_fallback')
                except Exception as e2:
                    st.error("Could not create alternative visualization either. Please check your data.")
            
            # Add interpretation
            with st.expander("📊 Interpretation"):
                st.markdown("""
                ### Understanding 3D Relationships
                
                The 3D scatter plot visualizes the combined effect of IR, LOI, and sample size (represented by
                marker size) on total project cost, providing a holistic view of how these factors interact.
                
                ### Key Insights
                
                1. **Complex Relationships**: The 3D plot reveals that project costs are influenced by the
                   interaction of multiple factors, not just individual variables in isolation.
                
                2. **Scale Effect**: Marker sizes represent sample sizes, showing how larger projects (bigger
                   markers) relate to other factors - revealing that high-cost projects often have lower IRs
                   and higher LOIs.
                
                3. **Won vs. Lost Clustering**: The spatial distribution of won vs. lost bids in the 3D space
                   reveals pricing patterns and competitive thresholds across different combinations of factors.
                
                4. **Total Cost Perspective**: While CPI is important, this view brings focus to total project
                   cost, which is ultimately what matters for budget decisions and overall competitiveness.
                """)
    
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error in analysis component: {error_details}")
        st.error(f"An error occurred while rendering the analysis component: {str(e)}")
        st.error("Please try adjusting your filters or contact support if the problem persists.")