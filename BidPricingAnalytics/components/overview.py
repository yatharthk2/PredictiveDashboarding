"""
Overview component for the CPI Analysis & Prediction Dashboard.
Displays a high-level summary of the data and key metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Any, Optional

# Import visualization utilities
from utils.visualization import (
    create_type_distribution_chart,
    create_cpi_distribution_boxplot,
    create_cpi_vs_ir_scatter,
    create_cpi_efficiency_chart
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def show_overview(won_data: pd.DataFrame, lost_data: pd.DataFrame, combined_data: pd.DataFrame) -> None:
    """
    Display the overview dashboard showing key metrics and charts.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
        combined_data (pd.DataFrame): Combined DataFrame of won and lost bids
    """
    try:
        st.title("CPI Analysis Dashboard: Overview")
        
        # Introduction with help tooltips
        st.markdown("""
        This dashboard analyzes the Cost Per Interview (CPI) between won and lost bids 
        to identify meaningful differences. The three main factors that influence CPI are:
        
        - **IR (Incidence Rate)**: The percentage of people who qualify for a survey
        - **LOI (Length of Interview)**: How long the survey takes to complete
        - **Sample Size (Completes)**: The number of completed surveys
        
        Use the navigation menu on the left to explore different analyses and tools.
        """)
        
        # Help expander with more details
        with st.expander("📖 How to use this dashboard"):
            st.markdown("""
            ### Dashboard Guide
            
            This overview page provides a high-level summary of your CPI data, showing the key differences
            between won and lost bids. Here's how to make the most of it:
            
            1. **Key Metrics Section**: Shows the average values for CPI, IR, and LOI, helping you quickly understand the differences between won and lost bids.
            
            2. **Data Distribution Charts**: Visualizes how won and lost bids are distributed, providing context for your analysis.
            
            3. **CPI vs. Factors Charts**: Shows how CPI relates to other factors, helping identify patterns and relationships.
            
            4. **Navigation**: Use the sidebar to switch between different dashboard sections for deeper analysis.
            
            5. **Filtering**: Toggle the "Filter out extreme values" option in the sidebar to include or exclude outliers.
            
            For more detailed analysis, use the CPI Analysis, CPI Prediction, and Insights sections accessible from the sidebar.
            """)
        
        # Key metrics
        st.header("Key Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Average CPI - Won", 
                f"${won_data['CPI'].mean():.2f}"
            )
            st.metric(
                "Average CPI - Lost", 
                f"${lost_data['CPI'].mean():.2f}"
            )
            st.metric(
                "CPI Difference", 
                f"${lost_data['CPI'].mean() - won_data['CPI'].mean():.2f}",
                delta=f"{((lost_data['CPI'].mean() - won_data['CPI'].mean()) / won_data['CPI'].mean() * 100):.1f}%"
            )
        
        with col2:
            st.metric(
                "Average IR - Won", 
                f"{won_data['IR'].mean():.2f}%"
            )
            st.metric(
                "Average IR - Lost", 
                f"{lost_data['IR'].mean():.2f}%"
            )
            ir_diff = lost_data['IR'].mean() - won_data['IR'].mean()
            st.metric(
                "IR Difference", 
                f"{ir_diff:.2f}%",
                delta=f"{ir_diff:.2f}%"
            )
        
        with col3:
            st.metric(
                "Average LOI - Won", 
                f"{won_data['LOI'].mean():.2f} min"
            )
            st.metric(
                "Average LOI - Lost", 
                f"{lost_data['LOI'].mean():.2f} min"
            )
            loi_diff = lost_data['LOI'].mean() - won_data['LOI'].mean()
            st.metric(
                "LOI Difference", 
                f"{loi_diff:.2f} min",
                delta=f"{loi_diff:.2f} min"
            )
        
        # Overview charts
        st.header("Data Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create and display pie chart
            fig = create_type_distribution_chart(combined_data)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Create and display CPI boxplot
            fig = create_cpi_distribution_boxplot(won_data, lost_data)
            st.plotly_chart(fig, use_container_width=True)
        
        # CPI vs IR scatter plot
        st.header("CPI vs Key Factors")
        
        # IR vs CPI relationship
        st.subheader("CPI vs Incidence Rate (IR)")
        fig = create_cpi_vs_ir_scatter(won_data, lost_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interpretation
        with st.expander("📊 Interpretation"):
            st.markdown("""
            ### What This Chart Shows
            
            The scatter plot visualizes how CPI (Cost Per Interview) relates to Incidence Rate (IR). 
            Each dot represents a project, with blue dots showing won bids and orange dots showing lost bids.
            
            ### Key Insights
            
            1. **Inverse Relationship**: Generally, as IR increases, CPI decreases. This makes sense because higher
               incidence rates mean it's easier to find qualified respondents.
            
            2. **Won vs Lost Gap**: Notice that lost bids (orange) tend to have higher CPIs than won bids (blue)
               at similar IR levels. This suggests that pricing competitiveness is a key factor in winning bids.
            
            3. **Trend Lines**: The lines show the general trend for each bid type. The gap between these lines
               represents the typical pricing differential between won and lost bids.
            """)
        
        # Add CPI efficiency chart (new visualization)
        st.subheader("CPI Efficiency Analysis")
        fig = create_cpi_efficiency_chart(won_data, lost_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interpretation
        with st.expander("📊 Interpretation"):
            st.markdown("""
            ### What This Chart Shows
            
            This chart visualizes a combined "efficiency metric" that incorporates IR, LOI, and sample size
            into a single value. Higher efficiency values indicate more favorable project parameters.
            
            ### Key Insights
            
            1. **Efficiency Correlation**: There's a relationship between the efficiency metric and CPI,
               with more efficient projects (higher value) generally having lower CPIs.
            
            2. **Won vs Lost Comparison**: Won bids tend to show better efficiency-to-price ratios than lost bids,
               suggesting that competitive pricing aligned with project parameters is important for winning bids.
            
            3. **Trend Analysis**: The trend lines show how CPI typically scales with efficiency for each bid type,
               helping to identify optimal pricing points based on project parameters.
            """)
        
        # Recent trends section
        st.header("Project Volume Trends")
        
        # Add a date filter if there's date data available
        date_col = None
        if 'Date' in won_data.columns:
            date_col = 'Date'
        elif 'Project Date' in won_data.columns:
            date_col = 'Project Date'
        elif 'Invoiced Date' in won_data.columns:
            date_col = 'Invoiced Date'
        
        if date_col is not None:
            # Convert date column to datetime if it's not already
            try:
                won_data[date_col] = pd.to_datetime(won_data[date_col])
                
                # Create a date range for the last 12 months
                current_date = won_data[date_col].max()
                one_year_ago = current_date - pd.DateOffset(months=12)
                
                # Filter data for the last 12 months
                recent_won = won_data[won_data[date_col] >= one_year_ago]
                
                # Group by month and count projects
                if not recent_won.empty:
                    recent_won['Month'] = recent_won[date_col].dt.to_period('M')
                    monthly_counts = recent_won.groupby('Month').size().reset_index(name='Count')
                    monthly_counts['Month'] = monthly_counts['Month'].astype(str)
                    
                    # Create bar chart
                    fig = px.bar(
                        monthly_counts,
                        x='Month',
                        y='Count',
                        title='Monthly Project Volume (Last 12 Months)',
                        labels={'Count': 'Number of Projects', 'Month': ''},
                        color_discrete_sequence=['#3288bd']  # Blue, colorblind friendly
                    )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis=dict(tickangle=45),
                        yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
                        plot_bgcolor='rgba(255,255,255,1)',
                        paper_bgcolor='rgba(255,255,255,1)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No project data available for the last 12 months.")
            except Exception as e:
                logger.error(f"Error creating date trends: {e}")
                st.info("Could not analyze project date trends due to data format issues.")
        else:
            st.info("Date information not available for trend analysis.")
        
        # Add footer with call to action
        st.markdown("---")
        st.markdown("""
        **Next Steps**: Explore the **CPI Analysis** section for more detailed breakdowns by Incidence Rate, 
        Length of Interview, and Sample Size. Or try the **CPI Prediction** tool to find optimal pricing 
        for new projects.
        """)
    
    except Exception as e:
        logger.error(f"Error in overview component: {e}", exc_info=True)
        st.error(f"An error occurred while rendering the overview component: {str(e)}")
