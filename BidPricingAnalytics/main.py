"""
CPI Analysis & Prediction Dashboard
Main application file that orchestrates the dashboard components.

This application provides a comprehensive analysis of Cost Per Interview (CPI)
for market research projects, including visualization, analysis, and prediction tools.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Tuple, Any, Optional

# Import configuration
import config

# Import utility modules
from utils.data_loader import load_data
from utils.data_processor import apply_all_bins, engineer_features, get_data_summary

# Import components
from components.overview import show_overview
from components.analysis import show_analysis
from components.prediction import show_prediction
from components.insights import show_insights

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout=config.APP_LAYOUT,
    initial_sidebar_state=config.SIDEBAR_STATE
)

def main():
    """Main application function to orchestrate the dashboard."""
    # Add app title and description
    st.sidebar.title("CPI Analysis & Prediction")
    
    # Navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Choose a mode",
        ["Overview", "CPI Analysis", "CPI Prediction", "Insights & Recommendations"]
    )
    
    # Load data
    with st.spinner("Loading data..."):
        try:
            data = load_data()
            won_df = data['won']
            won_df_filtered = data['won_filtered']
            lost_df = data['lost']
            lost_df_filtered = data['lost_filtered']
            combined_df = data['combined']
            combined_df_filtered = data['combined_filtered']
            
            # Log data shapes
            logger.info(f"Won deals: {won_df.shape}")
            logger.info(f"Lost deals: {lost_df.shape}")
            logger.info(f"Combined: {combined_df.shape}")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.error("Please check that all required data files are in the correct location.")
            st.stop()
    
    # Process data - add bins to all dataframes
    try:
        won_df = apply_all_bins(won_df)
        won_df_filtered = apply_all_bins(won_df_filtered)
        lost_df = apply_all_bins(lost_df)
        lost_df_filtered = apply_all_bins(lost_df_filtered)
        combined_df = apply_all_bins(combined_df)
        combined_df_filtered = apply_all_bins(combined_df_filtered)
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.stop()
    
    # Add sidebar filters
    st.sidebar.title("Filtering Options")
    
    # Filter for extreme values
    show_filtered = st.sidebar.checkbox(
        "Filter out extreme values (>95th percentile)", 
        value=True,
        help="Remove outliers with very high CPI values to focus on typical cases"
    )
    
    # Choose datasets based on filtering option
    if show_filtered:
        won_data = won_df_filtered
        lost_data = lost_df_filtered
        combined_data = combined_df_filtered
    else:
        won_data = won_df
        lost_data = lost_df
        combined_data = combined_df
    
    # Display metrics
    data_summary = get_data_summary(combined_data)
    
    st.sidebar.title("Data Summary")
    if 'Won' in data_summary:
        st.sidebar.metric(
            "Won Bids Avg CPI", 
            f"${data_summary['Won']['Avg_CPI']:.2f}"
        )
    
    if 'Lost' in data_summary:
        st.sidebar.metric(
            "Lost Bids Avg CPI", 
            f"${data_summary['Lost']['Avg_CPI']:.2f}"
        )
    
    if 'Won' in data_summary and 'Lost' in data_summary:
        diff = data_summary['Lost']['Avg_CPI'] - data_summary['Won']['Avg_CPI']
        st.sidebar.metric(
            "CPI Difference", 
            f"${diff:.2f}",
            delta=f"{diff:.2f}"
        )
    
    # Add footer with info
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This dashboard provides analysis and prediction tools for "
        "Cost Per Interview (CPI) pricing in market research projects. "
        "Navigate between different sections using the radio buttons above."
    )
    
    # Show selected component based on app_mode
    if app_mode == "Overview":
        show_overview(won_data, lost_data, combined_data)
    
    elif app_mode == "CPI Analysis":
        show_analysis(won_data, lost_data, combined_data)
    
    elif app_mode == "CPI Prediction":
        # Engineer features for the prediction model
        combined_data_engineered = engineer_features(combined_data)
        show_prediction(combined_data_engineered, won_data, lost_data)
    
    elif app_mode == "Insights & Recommendations":
        show_insights(won_data, lost_data, combined_data)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logger.error(f"Unexpected error in main application: {e}", exc_info=True)
