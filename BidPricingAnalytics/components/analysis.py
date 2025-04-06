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

# Import visualization utilities
from utils.visualization import (
    create_cpi_histogram_comparison,
    create_cpi_vs_ir_scatter,
    create_bar_chart_by_bin,
    create_heatmap
)

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
            fig = create_cpi_histogram_comparison(won_data, lost_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add CPI statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Won Bids CPI Statistics")
                st.markdown(f"""
                - **Minimum:** ${won_data['CPI'].min():.2f}
                - **Maximum:** ${won_data['CPI'].max():.2f}
                - **Mean:** ${won_data['CPI'].mean():.2f}
                - **Median:** ${won_data['CPI'].median():.2f}
                - **Standard Deviation:** ${won_data['CPI'].std():.2f}
                """)
            
            with col2:
                st.subheader("Lost Bids CPI Statistics")
                st.markdown(f"""
                - **Minimum:** ${lost_data['CPI'].min():.2f}
                - **Maximum:** ${lost_data['CPI'].max():.2f}
                - **Mean:** ${lost_data['CPI'].mean():.2f}
                - **Median:** ${lost_data['CPI'].median():.2f}
                - **Standard Deviation:** ${lost_data['CPI'].std():.2f}
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
            fig = create_cpi_vs_ir_scatter(won_data, lost_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Create and display CPI by IR Bin bar chart
            st.subheader("Average CPI by IR Bin")
            fig = create_bar_chart_by_bin(won_data, lost_data, 'IR_Bin', 'CPI',
                                        title='Average CPI by Incidence Rate Bin')
            st.plotly_chart(fig, use_container_width=True)
            
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
            
            # Create CPI vs LOI scatter plot
            fig = go.Figure()
            
            # Add Won data
            fig.add_trace(go.Scatter(
                x=won_data['LOI'], 
                y=won_data['CPI'], 
                mode='markers',
                marker=dict(color='#3288bd', size=8, opacity=0.6),
                name="Won",
                hovertemplate='<b>Won Bid</b><br>LOI: %{x:.1f} min<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<extra></extra>',
                customdata=won_data[['IR']]
            ))
            
            # Add Lost data
            fig.add_trace(go.Scatter(
                x=lost_data['LOI'], 
                y=lost_data['CPI'], 
                mode='markers',
                marker=dict(color='#f58518', size=8, opacity=0.6),
                name="Lost",
                hovertemplate='<b>Lost Bid</b><br>LOI: %{x:.1f} min<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<extra></extra>',
                customdata=lost_data[['IR']]
            ))
            
            # Add trend lines
            # Won trend line
            won_trend_x = np.linspace(won_data['LOI'].min(), won_data['LOI'].max(), 100)
            won_coeffs = np.polyfit(won_data['LOI'], won_data['CPI'], 1)
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
            lost_trend_x = np.linspace(lost_data['LOI'].min(), lost_data['LOI'].max(), 100)
            lost_coeffs = np.polyfit(lost_data['LOI'], lost_data['CPI'], 1)
            lost_trend_y = np.polyval(lost_coeffs, lost_trend_x)
            
            fig.add_trace(go.Scatter(
                x=lost_trend_x,
                y=lost_trend_y,
                mode='lines',
                line=dict(color='#f58518', width=2),
                name='Lost Trend',
                hoverinfo='skip'
            ))
            
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
                yaxis=dict(gridcolor='rgba(0,0,0,0.1)', tickprefix=')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create and display CPI by LOI Bin bar chart
            st.subheader("Average CPI by LOI Bin")
            fig = create_bar_chart_by_bin(won_data, lost_data, 'LOI_Bin', 'CPI',
                                        title='Average CPI by Length of Interview Bin')
            st.plotly_chart(fig, use_container_width=True)
            
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
            
            # Create CPI vs Completes scatter plot
            fig = go.Figure()
            
            # Add Won data
            fig.add_trace(go.Scatter(
                x=won_data['Completes'], 
                y=won_data['CPI'], 
                mode='markers',
                marker=dict(color='#3288bd', size=8, opacity=0.6),
                name="Won",
                hovertemplate='<b>Won Bid</b><br>Completes: %{x}<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<br>LOI: %{customdata[1]:.1f} min<extra></extra>',
                customdata=won_data[['IR', 'LOI']]
            ))
            
            # Add Lost data
            fig.add_trace(go.Scatter(
                x=lost_data['Completes'], 
                y=lost_data['CPI'], 
                mode='markers',
                marker=dict(color='#f58518', size=8, opacity=0.6),
                name="Lost",
                hovertemplate='<b>Lost Bid</b><br>Completes: %{x}<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]:.1f}%<br>LOI: %{customdata[1]:.1f} min<extra></extra>',
                customdata=lost_data[['IR', 'LOI']]
            ))
            
            # Add trend lines using logarithmic fit (better for sample size relationships)
            # Won trend
            won_x = won_data['Completes']
            won_y = won_data['CPI']
            won_log_x = np.log(won_x)
            won_coeffs = np.polyfit(won_log_x, won_y, 1)
            
            won_trend_x = np.linspace(won_x.min(), won_x.max(), 100)
            won_trend_y = won_coeffs[0] * np.log(won_trend_x) + won_coeffs[1]
            
            fig.add_trace(go.Scatter(
                x=won_trend_x,
                y=won_trend_y,
                mode='lines',
                line=dict(color='#3288bd', width=2),
                name='Won Trend',
                hoverinfo='skip'
            ))
            
            # Lost trend
            lost_x = lost_data['Completes']
            lost_y = lost_data['CPI']
            lost_log_x = np.log(lost_x)
            lost_coeffs = np.polyfit(lost_log_x, lost_y, 1)
            
            lost_trend_x = np.linspace(lost_x.min(), lost_x.max(), 100)
            lost_trend_y = lost_coeffs[0] * np.log(lost_trend_x) + lost_coeffs[1]
            
            fig.add_trace(go.Scatter(
                x=lost_trend_x,
                y=lost_trend_y,
                mode='lines',
                line=dict(color='#f58518', width=2),
                name='Lost Trend',
                hoverinfo='skip'
            ))
            
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
                yaxis=dict(gridcolor='rgba(0,0,0,0.1)', tickprefix=')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create and display CPI by Completes Bin bar chart
            st.subheader("Average CPI by Sample Size Bin")
            fig = create_bar_chart_by_bin(won_data, lost_data, 'Completes_Bin', 'CPI',
                                        title='Average CPI by Sample Size Bin')
            st.plotly_chart(fig, use_container_width=True)
            
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
            
            # Create pivot tables for heatmaps
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
            
            # Display heatmaps
            st.subheader("IR and LOI Combined Influence on CPI")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Won deals heatmap
                fig = create_heatmap(
                    won_pivot, 
                    "Won Deals: Average CPI by IR and LOI",
                    HEATMAP_COLORSCALE_WON
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Lost deals heatmap
                fig = create_heatmap(
                    lost_pivot, 
                    "Lost Deals: Average CPI by IR and LOI",
                    HEATMAP_COLORSCALE_LOST
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Create a differential heatmap
            st.subheader("CPI Differential: Lost vs. Won")
            
            # Calculate CPI differential
            diff_pivot = lost_pivot - won_pivot
            
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
            
            st.plotly_chart(fig, use_container_width=True)
            
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
            
            # Additional multi-factor analysis
            st.subheader("Combined Factor Impact on Total Project Cost")
            
            # Calculate total project cost
            won_data['Total_Cost'] = won_data['CPI'] * won_data['Completes']
            lost_data['Total_Cost'] = lost_data['CPI'] * lost_data['Completes']
            
            # Create 3D scatter plot
            fig = go.Figure()
            
            # Add Won data
            fig.add_trace(go.Scatter3d(
                x=won_data['IR'],
                y=won_data['LOI'],
                z=won_data['Total_Cost'],
                mode='markers',
                marker=dict(
                    size=won_data['Completes'] / 50,  # Size based on completes
                    color='#3288bd',
                    opacity=0.7,
                    symbol='circle'
                ),
                name='Won',
                hovertemplate='<b>Won Bid</b><br>IR: %{x:.1f}%<br>LOI: %{y:.1f} min<br>Total Cost: $%{z:.2f}<br>Completes: %{customdata[0]}<br>CPI: $%{customdata[1]:.2f}<extra></extra>',
                customdata=won_data[['Completes', 'CPI']]
            ))
            
            # Add Lost data
            fig.add_trace(go.Scatter3d(
                x=lost_data['IR'],
                y=lost_data['LOI'],
                z=lost_data['Total_Cost'],
                mode='markers',
                marker=dict(
                    size=lost_data['Completes'] / 50,  # Size based on completes
                    color='#f58518',
                    opacity=0.7,
                    symbol='circle'
                ),
                name='Lost',
                hovertemplate='<b>Lost Bid</b><br>IR: %{x:.1f}%<br>LOI: %{y:.1f} min<br>Total Cost: $%{z:.2f}<br>Completes: %{customdata[0]}<br>CPI: $%{customdata[1]:.2f}<extra></extra>',
                customdata=lost_data[['Completes', 'CPI']]
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
            
            st.plotly_chart(fig, use_container_width=True)
            
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
        logger.error(f"Error in analysis component: {e}", exc_info=True)
        st.error(f"An error occurred while rendering the analysis component: {str(e)}")
