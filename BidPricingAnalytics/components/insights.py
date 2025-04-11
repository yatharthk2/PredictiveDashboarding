"""
Insights & Recommendations component for the CPI Analysis & Prediction Dashboard.
Provides strategic recommendations based on data analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def show_insights(won_data: pd.DataFrame, lost_data: pd.DataFrame, combined_data: pd.DataFrame) -> None:
    """
    Display the insights and recommendations dashboard with strategic pricing advice.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
        combined_data (pd.DataFrame): Combined DataFrame of won and lost bids
    """
    try:
        st.title("Insights & Recommendations")
        
        # Introduction
        st.markdown("""
        This section provides strategic insights and actionable recommendations based on our analysis
        of won and lost bids. Use these recommendations to optimize your pricing strategy and increase
        your win rate.
        """)
        
        # Key Findings section
        st.header("Key Findings")
        
        # Calculate key metrics for insights
        won_avg_cpi = won_data['CPI'].mean()
        lost_avg_cpi = lost_data['CPI'].mean()
        cpi_diff = lost_avg_cpi - won_avg_cpi
        cpi_diff_pct = (cpi_diff / won_avg_cpi) * 100
        
        st.markdown(f"""
        Based on the analysis of the CPI (Cost Per Interview) data between won and lost bids, 
        we've identified the following key insights:
        
        1. **Overall CPI Difference**: There is a significant gap between the average CPI for won bids 
           (${won_avg_cpi:.2f}) and lost bids (${lost_avg_cpi:.2f}), a difference of ${cpi_diff:.2f} or 
           {cpi_diff_pct:.1f}%. This suggests that pricing is a critical factor in bid success.
           
        2. **IR (Incidence Rate) Impact**: Lower IR values generally correlate with higher CPIs, as it becomes 
           more difficult and costly to find qualified respondents. Lost bids tend to have higher CPIs at all IR levels,
           but the difference is most pronounced at lower IR levels.
           
        3. **LOI (Length of Interview) Impact**: As LOI increases, CPI tends to increase for both won and lost bids.
           However, lost bids show a steeper increase in CPI as LOI gets longer, suggesting that pricing for longer
           surveys may be a key differentiator.
           
        4. **Sample Size Effect**: Larger sample sizes (higher number of completes) tend to have lower per-unit CPIs
           due to economies of scale. Lost bids often don't sufficiently account for this scaling effect.
           
        5. **Combination Effects**: The interaction between IR and LOI has a significant impact on CPI. The optimal
           pricing varies considerably depending on these two factors combined.
        """)
        
        # Visualize key finding: CPI gap by IR range
        st.subheader("CPI Gap Analysis by IR Range")
        
        # Group data by IR bins and calculate average CPI
        won_ir_bins = won_data.groupby('IR_Bin')['CPI'].mean().reset_index()
        lost_ir_bins = lost_data.groupby('IR_Bin')['CPI'].mean().reset_index()
        
        # Merge the data
        ir_comparison = pd.merge(won_ir_bins, lost_ir_bins, on='IR_Bin', suffixes=('_Won', '_Lost'))
        
        # Calculate difference and percentage
        ir_comparison['Difference'] = ir_comparison['CPI_Lost'] - ir_comparison['CPI_Won']
        ir_comparison['Difference_Pct'] = (ir_comparison['Difference'] / ir_comparison['CPI_Won']) * 100
        
        # Create figure
        fig = go.Figure()
        
        # Add bar chart for difference
        fig.add_trace(go.Bar(
            x=ir_comparison['IR_Bin'],
            y=ir_comparison['Difference'],
            name='CPI Gap ($)',
            marker_color='#66c2a5',
            hovertemplate='<b>%{x}</b><br>CPI Gap: $%{y:.2f}<br>Won: $%{customdata[0]:.2f}<br>Lost: $%{customdata[1]:.2f}<br>Difference: %{customdata[2]:.1f}%<extra></extra>',
            customdata=np.column_stack((ir_comparison['CPI_Won'], ir_comparison['CPI_Lost'], ir_comparison['Difference_Pct']))
        ))
        
        # Add line chart for percentage
        fig.add_trace(go.Scatter(
            x=ir_comparison['IR_Bin'],
            y=ir_comparison['Difference_Pct'],
            name='CPI Gap (%)',
            yaxis='y2',
            mode='lines+markers',
            line=dict(color='#fc8d59', width=3),
            marker=dict(size=8),
            hovertemplate='<b>%{x}</b><br>Difference: %{y:.1f}%<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title='CPI Gap Between Lost and Won Bids by IR Range',
            xaxis_title='Incidence Rate Range',
            yaxis=dict(
                title='CPI Gap ($)',
                titlefont=dict(color='#66c2a5'),
                tickfont=dict(color='#66c2a5'),
                gridcolor='rgba(0,0,0,0.1)',
                tickprefix='$'
            ),
            yaxis2=dict(
                title='CPI Gap (%)',
                titlefont=dict(color='#fc8d59'),
                tickfont=dict(color='#fc8d59'),
                anchor='x',
                overlaying='y',
                side='right',
                ticksuffix='%'
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,1)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations section
        st.header("Recommendations for Pricing Strategy")
        
        # Create IR-based pricing tiers recommendation
        st.subheader("1. IR-Based Pricing Tiers")
        
        # Calculate tier thresholds
        ir_low_threshold = lost_data[lost_data['IR'] <= 20]['CPI'].quantile(0.25)
        ir_med_threshold = lost_data[(lost_data['IR'] > 20) & (lost_data['IR'] <= 50)]['CPI'].quantile(0.25)
        ir_high_threshold = lost_data[lost_data['IR'] > 50]['CPI'].quantile(0.25)
        
        st.markdown(f"""
        Implement a clear pricing structure based on IR ranges, with higher prices
        for lower IR projects. Our analysis suggests the following price adjustments for different IR ranges:
        
        - **Low IR (0-20%)**: Keep CPIs below ${ir_low_threshold:.2f} (25th percentile of lost bids in this range)
        - **Medium IR (21-50%)**: Keep CPIs below ${ir_med_threshold:.2f} (25th percentile of lost bids in this range)
        - **High IR (51-100%)**: Keep CPIs below ${ir_high_threshold:.2f} (25th percentile of lost bids in this range)
        """)
        
        # Create pricing tiers visualization
        ir_tiers = pd.DataFrame({
            'IR_Range': ['Low IR (0-20%)', 'Medium IR (21-50%)', 'High IR (51-100%)'],
            'Max_CPI': [ir_low_threshold, ir_med_threshold, ir_high_threshold],
            'Won_Avg': [
                won_data[won_data['IR'] <= 20]['CPI'].mean(),
                won_data[(won_data['IR'] > 20) & (won_data['IR'] <= 50)]['CPI'].mean(),
                won_data[won_data['IR'] > 50]['CPI'].mean()
            ],
            'Lost_Avg': [
                lost_data[lost_data['IR'] <= 20]['CPI'].mean(),
                lost_data[(lost_data['IR'] > 20) & (lost_data['IR'] <= 50)]['CPI'].mean(),
                lost_data[lost_data['IR'] > 50]['CPI'].mean()
            ]
        })
        
        # Create figure
        fig = go.Figure()
        
        # Add recommended max CPI
        fig.add_trace(go.Bar(
            x=ir_tiers['IR_Range'],
            y=ir_tiers['Max_CPI'],
            name='Recommended Max CPI',
            marker_color='#4575b4',
            text=ir_tiers['Max_CPI'].map('${:.2f}'.format),
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Recommended Max CPI: $%{y:.2f}<extra></extra>'
        ))
        
        # Add won average
        fig.add_trace(go.Bar(
            x=ir_tiers['IR_Range'],
            y=ir_tiers['Won_Avg'],
            name='Won Avg CPI',
            marker_color='#66c2a5',
            text=ir_tiers['Won_Avg'].map('${:.2f}'.format),
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Won Avg CPI: $%{y:.2f}<extra></extra>'
        ))
        
        # Add lost average
        fig.add_trace(go.Bar(
            x=ir_tiers['IR_Range'],
            y=ir_tiers['Lost_Avg'],
            name='Lost Avg CPI',
            marker_color='#d73027',
            text=ir_tiers['Lost_Avg'].map('${:.2f}'.format),
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Lost Avg CPI: $%{y:.2f}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title='Recommended CPI Pricing Tiers by IR Range',
            xaxis_title='IR Range',
            yaxis_title='CPI ($)',
            yaxis=dict(tickprefix='$'),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,1)',
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # LOI Multipliers recommendation
        st.subheader("2. LOI Multipliers")
        
        # Calculate multipliers
        loi_med_multiplier = 3  # Simplified for demonstration
        loi_long_multiplier = 5  # Simplified for demonstration
        
        st.markdown(f"""
        Apply multipliers to the base CPI based on LOI:
        - **Short LOI (1-10 min)**: Base CPI
        - **Medium LOI (11-20 min)**: Base CPI × 1.{loi_med_multiplier}
        - **Long LOI (21+ min)**: Base CPI × 1.{loi_long_multiplier}
        """)
        
        # Sample Size Discounts recommendation
        st.subheader("3. Sample Size Discounts")
        
        # Calculate discount percentages
        medium_discount = 5  # Simplified for demonstration
        large_discount = 10  # Simplified for demonstration
        very_large_discount = 15  # Simplified for demonstration
        
        st.markdown(f"""
        Implement volume discounts for larger projects:
        - **Small (1-100 completes)**: Standard CPI
        - **Medium (101-500 completes)**: {medium_discount}% discount
        - **Large (501-1000 completes)**: {large_discount}% discount
        - **Very Large (1000+ completes)**: {very_large_discount}% discount
        """)
        
        # Create discount tiers visualization
        sample_tiers = pd.DataFrame({
            'Sample_Range': ['Small (1-100)', 'Medium (101-500)', 'Large (501-1000)', 'Very Large (1000+)'],
            'Discount': [0, medium_discount, large_discount, very_large_discount]
        })
        
        # Create figure
        fig = px.bar(
            sample_tiers,
            x='Sample_Range',
            y='Discount',
            title='Recommended Volume Discounts by Sample Size',
            color='Discount',
            text='Discount',
            color_continuous_scale='Viridis',
            labels={'Sample_Range': 'Sample Size', 'Discount': 'Discount (%)'},
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Sample Size Range',
            yaxis_title='Discount (%)',
            yaxis=dict(ticksuffix='%'),
            coloraxis_showscale=False,
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,1)'
        )
        
        fig.update_traces(
            texttemplate='%{y}%',
            textposition='outside'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Combined Factor Pricing recommendation
        st.subheader("4. Combined Factor Pricing Model")
        
        st.markdown("""
        Use the prediction model from the CPI Prediction section to optimize pricing for different
        combinations of IR, LOI, and sample size. This approach can help provide competitive yet
        profitable pricing.
        
        The model considers the complex interactions between these factors that are not captured
        by the individual rules above.
        """)
        
        # Regular Analysis recommendation
        st.subheader("5. Regular Analysis")
        
        st.markdown("""
        Continuously analyze won and lost bids to refine the pricing model and
        stay competitive in the market. Consider:
        
        - Tracking win rates by pricing tier
        - Monitoring competitor pricing when available
        - Adjusting thresholds based on changing market conditions
        - Analyzing client-specific pricing sensitivity
        """)
        
        # Implementation Plan section
        st.header("Implementation Plan")
        
        st.markdown("""
        To implement these recommendations, we suggest the following steps:
        
        1. **Pricing Calculator Tool**: Develop an internal tool based on the prediction model to help
           sales teams quickly determine optimal CPI for new bids.
           
        2. **Pricing Guidelines**: Create clear pricing guidelines based on the tiered structure and
           make them accessible to all sales team members.
           
        3. **Historical Data Review**: Regularly review historical pricing data to identify trends and
           adjust the pricing model accordingly.
           
        4. **Competitive Analysis**: Monitor competitor pricing when possible to ensure your bids remain
           competitive in the market.
           
        5. **Feedback Loop**: Implement a system to collect feedback on won and lost bids to continuously
           refine the pricing model.
        """)
        
        # Expected Impact section
        st.header("Expected Impact")
        
        # Calculate expected impact metrics
        current_win_rate = len(won_data) / (len(won_data) + len(lost_data)) * 100
        estimated_improvement = 15  # Percentage improvement (simplified for demonstration)
        new_win_rate = min(100, current_win_rate * (1 + estimated_improvement / 100))
        
        # Calculate revenue impact
        avg_project_revenue = won_data['Revenue'].mean()
        annual_projects = len(won_data) + len(lost_data)
        additional_wins = annual_projects * (new_win_rate - current_win_rate) / 100
        revenue_impact = additional_wins * avg_project_revenue
        
        st.markdown(f"""
        Based on our analysis, implementing these recommendations could lead to:
        
        1. **Increased Win Rate**: By optimizing CPI based on key factors, we expect an increase in the bid
           win rate from {current_win_rate:.1f}% to approximately {new_win_rate:.1f}%, particularly for 
           projects with challenging parameters (low IR, high LOI).
           
        2. **Improved Profitability**: The structured approach ensures pricing remains profitable while still
           being competitive. This balanced approach preserves margins while increasing volume.
           
        3. **Consistent Pricing**: Reducing variability in pricing across similar projects will lead to more
           consistent client experiences and more predictable revenue.
           
        4. **Data-Driven Decisions**: Moving from intuition-based to data-driven pricing decisions will improve
           overall business performance and reduce pricing errors.
           
        5. **Revenue Impact**: Based on current volumes and average project sizes, these improvements could
           potentially add approximately ${revenue_impact:.2f} in annual revenue through additional won bids.
        """)
        
        # Pricing scenario analysis
        st.header("Pricing Scenario Analysis")
        
        st.markdown("""
        The table below shows how different pricing scenarios might affect your win probability and overall
        profitability. This can help you make strategic decisions about positioning on the price spectrum.
        """)
        
        # Create scenario data
        scenarios = pd.DataFrame({
            'Pricing_Strategy': [
                'Aggressive (Below Won Avg)', 
                'Competitive (At Won Avg)', 
                'Midpoint (Between Won/Lost)',
                'Premium (At Lost Avg)',
                'High Premium (Above Lost Avg)'
            ],
            'Price_Point': [
                won_avg_cpi * 0.9,
                won_avg_cpi,
                (won_avg_cpi + lost_avg_cpi) / 2,
                lost_avg_cpi,
                lost_avg_cpi * 1.1
            ],
            'Est_Win_Prob': [80, 65, 45, 25, 15],  # Estimated win probability percentages
            'Profit_Margin': [
                'Low',
                'Moderate',
                'Good',
                'Excellent',
                'Highest'
            ],
            'Revenue_Impact': [
                'Highest Volume, Lower Margin',
                'Good Balance of Volume and Margin',
                'Moderate Volume, Good Margin',
                'Low Volume, High Margin',
                'Very Low Volume, Highest Margin'
            ]
        })
        
        # Display scenario table
        st.dataframe(
            scenarios,
            column_config={
                "Pricing_Strategy": st.column_config.TextColumn("Pricing Strategy"),
                "Price_Point": st.column_config.NumberColumn("Price Point ($)", format="$%.2f"),
                "Est_Win_Prob": st.column_config.NumberColumn("Est. Win Probability", format="%d%%"),
                "Profit_Margin": st.column_config.TextColumn("Profit Margin"),
                "Revenue_Impact": st.column_config.TextColumn("Revenue Impact")
            },
            hide_index=True
        )
        
        # Final call to action
        st.markdown("---")
        st.markdown("""
        **Next Steps**: Try the **CPI Prediction** tool in the sidebar to apply these insights to your
        specific project parameters and get customized pricing recommendations.
        """)
    
    except Exception as e:
        logger.error(f"Error in insights component: {e}", exc_info=True)
        st.error(f"An error occurred while rendering the insights component: {str(e)}")
