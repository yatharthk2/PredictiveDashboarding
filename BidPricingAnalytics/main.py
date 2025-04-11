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
import traceback
from typing import Dict, List, Tuple, Any, Optional

# Import configuration
import config

# Import utility modules
from utils.data_loader import load_data
from utils.data_processor import apply_all_bins, engineer_features, get_data_summary, detect_data_issues

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
    
    # Load data with improved error handling
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
            error_details = traceback.format_exc()
            logger.error(f"Error loading data: {error_details}")
            
            st.error(f"Error loading data: {str(e)}")
            st.error("Please check that all required data files are in the correct location.")
            
            # Try to provide more helpful error details
            try:
                # Check if required files exist
                file_status = {}
                for file_path in [config.INVOICED_JOBS_FILE, config.LOST_DEALS_FILE, config.ACCOUNT_SEGMENT_FILE]:
                    file_status[os.path.basename(file_path)] = os.path.exists(file_path)
                
                # Show file status
                st.warning("File status check:")
                for file_name, exists in file_status.items():
                    st.warning(f"- {file_name}: {'✅ Found' if exists else '❌ Missing'}")
                
                # Provide suggestion based on error
                if "No such file or directory" in str(e):
                    st.info("Suggestion: Check that your data files are in the correct location and have the correct names.")
                elif "cannot convert string to float" in str(e) or "could not convert string to float" in str(e):
                    st.info("Suggestion: There may be non-numeric values in columns that should be numeric. Check your data files.")
                elif "NULL" in str(e) or "null" in str(e):
                    st.info("Suggestion: Your data may contain NULL values that need to be handled. Try preprocessing your data.")
            except Exception:
                pass
            
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
        error_details = traceback.format_exc()
        logger.error(f"Error processing data: {error_details}")
        
        st.error(f"Error processing data: {str(e)}")
        st.error("This may be due to unexpected data formats or missing values.")
        st.stop()
    
    # Add sidebar filters
    st.sidebar.title("Filtering Options")
    
    # Filter for extreme values
    show_filtered = st.sidebar.checkbox(
        "Filter out extreme values (>95th percentile)", 
        value=True,
        help="Remove outliers with very high CPI values to focus on typical cases"
    )
    
    # Add extra filtering options
    with st.sidebar.expander("Advanced Filtering", expanded=False):
        # Add IR range filter
        ir_min = combined_df['IR'].min() if not np.isnan(combined_df['IR'].min()) else 0
        ir_max = combined_df['IR'].max() if not np.isnan(combined_df['IR'].max()) else 100
        ir_range = st.slider(
            "IR Range (%)",
            min_value=float(ir_min),
            max_value=float(ir_max),
            value=(float(ir_min), float(ir_max)),
            step=1.0,
            help="Filter by Incidence Rate range"
        )
        
        # Add LOI range filter
        loi_min = combined_df['LOI'].min() if not np.isnan(combined_df['LOI'].min()) else 0
        loi_max = combined_df['LOI'].max() if not np.isnan(combined_df['LOI'].max()) else 60
        loi_range = st.slider(
            "LOI Range (minutes)",
            min_value=float(loi_min),
            max_value=float(loi_max),
            value=(float(loi_min), float(loi_max)),
            step=1.0,
            help="Filter by Length of Interview range"
        )
        
        # Add CPI range filter
        cpi_min = combined_df['CPI'].min() if not np.isnan(combined_df['CPI'].min()) else 0
        cpi_max = combined_df['CPI'].max() if not np.isnan(combined_df['CPI'].max()) else 100
        cpi_range = st.slider(
            "CPI Range ($)",
            min_value=float(cpi_min),
            max_value=float(cpi_max),
            value=(float(cpi_min), float(cpi_max)),
            step=1.0,
            help="Filter by CPI range"
        )
        
        # Add segment filter if segments exist
        if 'Segment' in combined_df.columns:
            segments = combined_df['Segment'].unique()
            if len(segments) > 1:  # Only show filter if more than one segment
                selected_segments = st.multiselect(
                    "Client Segments",
                    options=segments,
                    default=list(segments),
                    help="Filter by client segment"
                )
    
    # Apply advanced filters if they were shown and used
    if 'ir_range' in locals():
        # Apply IR filter
        won_df = won_df[(won_df['IR'] >= ir_range[0]) & (won_df['IR'] <= ir_range[1])]
        won_df_filtered = won_df_filtered[(won_df_filtered['IR'] >= ir_range[0]) & (won_df_filtered['IR'] <= ir_range[1])]
        lost_df = lost_df[(lost_df['IR'] >= ir_range[0]) & (lost_df['IR'] <= ir_range[1])]
        lost_df_filtered = lost_df_filtered[(lost_df_filtered['IR'] >= ir_range[0]) & (lost_df_filtered['IR'] <= ir_range[1])]
        combined_df = combined_df[(combined_df['IR'] >= ir_range[0]) & (combined_df['IR'] <= ir_range[1])]
        combined_df_filtered = combined_df_filtered[(combined_df_filtered['IR'] >= ir_range[0]) & (combined_df_filtered['IR'] <= ir_range[1])]
        
        # Apply LOI filter
        won_df = won_df[(won_df['LOI'] >= loi_range[0]) & (won_df['LOI'] <= loi_range[1])]
        won_df_filtered = won_df_filtered[(won_df_filtered['LOI'] >= loi_range[0]) & (won_df_filtered['LOI'] <= loi_range[1])]
        lost_df = lost_df[(lost_df['LOI'] >= loi_range[0]) & (lost_df['LOI'] <= loi_range[1])]
        lost_df_filtered = lost_df_filtered[(lost_df_filtered['LOI'] >= loi_range[0]) & (lost_df_filtered['LOI'] <= loi_range[1])]
        combined_df = combined_df[(combined_df['LOI'] >= loi_range[0]) & (combined_df['LOI'] <= loi_range[1])]
        combined_df_filtered = combined_df_filtered[(combined_df_filtered['LOI'] >= loi_range[0]) & (combined_df_filtered['LOI'] <= loi_range[1])]
        
        # Apply CPI filter
        won_df = won_df[(won_df['CPI'] >= cpi_range[0]) & (won_df['CPI'] <= cpi_range[1])]
        won_df_filtered = won_df_filtered[(won_df_filtered['CPI'] >= cpi_range[0]) & (won_df_filtered['CPI'] <= cpi_range[1])]
        lost_df = lost_df[(lost_df['CPI'] >= cpi_range[0]) & (lost_df['CPI'] <= cpi_range[1])]
        lost_df_filtered = lost_df_filtered[(lost_df_filtered['CPI'] >= cpi_range[0]) & (lost_df_filtered['CPI'] <= cpi_range[1])]
        combined_df = combined_df[(combined_df['CPI'] >= cpi_range[0]) & (combined_df['CPI'] <= cpi_range[1])]
        combined_df_filtered = combined_df_filtered[(combined_df_filtered['CPI'] >= cpi_range[0]) & (combined_df_filtered['CPI'] <= cpi_range[1])]
        
        # Apply segment filter if selected
        if 'selected_segments' in locals() and len(selected_segments) < len(segments):
            won_df = won_df[won_df['Segment'].isin(selected_segments)]
            won_df_filtered = won_df_filtered[won_df_filtered['Segment'].isin(selected_segments)]
            lost_df = lost_df[lost_df['Segment'].isin(selected_segments)]
            lost_df_filtered = lost_df_filtered[lost_df_filtered['Segment'].isin(selected_segments)]
            combined_df = combined_df[combined_df['Segment'].isin(selected_segments)]
            combined_df_filtered = combined_df_filtered[combined_df_filtered['Segment'].isin(selected_segments)]
    
    # Check if we have data after filtering
    for df_name, df in {
        'won_df': won_df,
        'won_df_filtered': won_df_filtered,
        'lost_df': lost_df,
        'lost_df_filtered': lost_df_filtered,
        'combined_df': combined_df,
        'combined_df_filtered': combined_df_filtered
    }.items():
        if len(df) == 0:
            st.warning(f"No data remains in {df_name} after applying filters. Try adjusting your filter settings.")
            # Reset to original data
            data = load_data()
            won_df = data['won']
            won_df_filtered = data['won_filtered']
            lost_df = data['lost']
            lost_df_filtered = data['lost_filtered']
            combined_df = data['combined']
            combined_df_filtered = data['combined_filtered']
            # Re-process data
            won_df = apply_all_bins(won_df)
            won_df_filtered = apply_all_bins(won_df_filtered)
            lost_df = apply_all_bins(lost_df)
            lost_df_filtered = apply_all_bins(lost_df_filtered)
            combined_df = apply_all_bins(combined_df)
            combined_df_filtered = apply_all_bins(combined_df_filtered)
            break
    
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
    
    # Add data quality indicators to sidebar
    with st.sidebar.expander("Data Quality", expanded=False):
        # Check data issues
        try:
            issues = detect_data_issues(combined_data)
            
            # Display issues summary
            if issues.get('extreme_outliers'):
                st.warning(f"Extreme outliers found in {len(issues['extreme_outliers'])} columns")
                for col, data in issues['extreme_outliers'].items():
                    st.warning(f"- {col}: {data['count']} values ({data['percentage']:.1f}%) above {data['threshold']:.2f}")
            else:
                st.success("No extreme outliers detected")
                
            if issues.get('collinearity', {}).get('high_correlation_pairs'):
                st.warning(f"High correlation detected between {len(issues['collinearity']['high_correlation_pairs'])} variable pairs")
                for pair in issues['collinearity']['high_correlation_pairs'][:3]:  # Show just the top 3
                    st.warning(f"- {pair['var1']} & {pair['var2']}: {pair['correlation']:.2f}")
            else:
                st.success("No high collinearity detected")
                
            if issues.get('missing_values'):
                st.warning(f"Missing values detected in {len(issues['missing_values'])} columns")
            else:
                st.success("No missing values detected")
        except Exception as e:
            st.warning(f"Could not analyze data quality: {str(e)}")
    
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
        try:
            # Apply feature engineering with progress indication
            with st.spinner("Preparing data for prediction..."):
                combined_data_engineered = engineer_features(combined_data)
            show_prediction(combined_data_engineered, won_data, lost_data)
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error in prediction preparation: {error_details}")
            st.error(f"Error preparing data for prediction: {str(e)}")
            st.error("This may be due to unexpected data patterns or extreme values.")
            
            # Offer suggestions
            st.warning("Suggestions:")
            st.warning("1. Try enabling the 'Filter out extreme values' option")
            st.warning("2. Check your data for unusual values or patterns")
            st.warning("3. Ensure numeric columns contain only valid numeric values")
    
    elif app_mode == "Insights & Recommendations":
        show_insights(won_data, lost_data, combined_data)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_details = traceback.format_exc()
        st.error(f"An unexpected error occurred: {str(e)}")
        logger.error(f"Unexpected error in main application: {error_details}")
        
        # Provide more helpful error context if possible
        error_str = str(e).lower()
        if "duplicate" in error_str and "key" in error_str:
            st.warning("This appears to be related to duplicate visualization elements. Try refreshing the page or restarting the application.")
        elif "svd" in error_str and "converge" in error_str:
            st.warning("This appears to be a mathematical convergence issue, possibly due to extreme outliers in the data. Try enabling more aggressive filtering.")
        elif "memory" in error_str:
            st.warning("This appears to be a memory-related issue. Try processing smaller subsets of your data or increase system memory.")
        
        # Add reporting option
        st.info("If this error persists, please share the error message and the steps that led to it with your development team.")