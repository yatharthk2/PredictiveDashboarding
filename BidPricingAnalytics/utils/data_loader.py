"""
Data loading functionality for the CPI Analysis & Prediction Dashboard.
Handles loading data from Excel files and initial preprocessing.
"""

import os
import pandas as pd
import streamlit as st
from typing import Dict, Any
import logging
from config import INVOICED_JOBS_FILE, LOST_DEALS_FILE, ACCOUNT_SEGMENT_FILE


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_data() -> Dict[str, pd.DataFrame]:
    """
    Load and process the data from Excel files.
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing different dataframes:
            - 'won': Won deals dataframe
            - 'won_filtered': Won deals with extreme values filtered out
            - 'lost': Lost deals dataframe
            - 'lost_filtered': Lost deals with extreme values filtered out
            - 'combined': Combined dataframe of won and lost deals
            - 'combined_filtered': Combined dataframe with extreme values filtered out
    
    Raises:
        FileNotFoundError: If any of the required data files are missing
        ValueError: If data processing fails due to unexpected data structure
    """
    try:
        # Check if files exist
        if not os.path.exists(INVOICED_JOBS_FILE):
            raise FileNotFoundError(f"Could not find the invoiced jobs file: {INVOICED_JOBS_FILE}")
        if not os.path.exists(LOST_DEALS_FILE):
            raise FileNotFoundError(f"Could not find the lost deals file: {LOST_DEALS_FILE}")
        
        # Load invoiced jobs data (Won deals)
        logger.info(f"Loading invoiced jobs data from {INVOICED_JOBS_FILE}")
        invoiced_df = pd.read_excel(INVOICED_JOBS_FILE)
        
        # Log column names for debugging
        logger.debug(f"Columns in invoiced_df: {invoiced_df.columns.tolist()}")
        
        # Rename columns to remove spaces
        invoiced_df = invoiced_df.rename(columns={
            ' CPI ': 'CPI',
            ' Actual Project Revenue ': 'Revenue',
            'Actual Project Revenue': 'Revenue',
            ' Revenue ': 'Revenue',
            'Revenue ': 'Revenue'
        })
        
        # Process Countries column
        invoiced_df['Countries'] = invoiced_df['Countries'].fillna('[]')
        invoiced_df['Country'] = invoiced_df['Countries'].apply(
            lambda x: x.replace('[', '').replace(']', '').replace('"', '')
        )
        invoiced_df['Country'] = invoiced_df['Country'].replace('', 'USA')
        
        # Create Won dataset
        won_df = invoiced_df[[
            'Project Code Parent', 'Client Name', 'CPI', 'Actual Ir', 'Actual Loi', 
            'Complete', 'Revenue', 'Invoiced Date', 'Country', 'Audience'
        ]].copy()
        
        # Rename columns for consistency
        won_df = won_df.rename(columns={
            'Project Code Parent': 'ProjectId',
            'Client Name': 'Client',
            'Actual Ir': 'IR',
            'Actual Loi': 'LOI',
            'Complete': 'Completes',
            'Invoiced Date': 'Date'
        })
        
        # Add type column
        won_df['Type'] = 'Won'
        
        # Load lost deals data
        logger.info(f"Loading lost deals data from {LOST_DEALS_FILE}")
        lost_df_raw = pd.read_excel(LOST_DEALS_FILE)
        
        # Filter for Sample items only
        lost_df = lost_df_raw[lost_df_raw['Item'] == 'Sample'].copy()
        
        # Create Lost dataset
        lost_df = lost_df[[
            'Record Id', 'Account Name', 'Customer Rate', 'IR', 'LOI', 
            'Qty', 'Item Amount', 'Description (Items)', 'Deal Name'
        ]].copy()
        
        # Rename columns for consistency
        lost_df = lost_df.rename(columns={
            'Record Id': 'DealId',
            'Account Name': 'Client',
            'Customer Rate': 'CPI',
            'Qty': 'Completes',
            'Item Amount': 'Revenue',
            'Description (Items)': 'Country',
            'Deal Name': 'ProjectName'
        })
        
        # Add type column
        lost_df['Type'] = 'Lost'
        
        # Try to load account segment data
        
        if os.path.exists(ACCOUNT_SEGMENT_FILE):
            logger.info(f"Loading account segment data from {ACCOUNT_SEGMENT_FILE}")
            try:
                client_segments = pd.read_csv(ACCOUNT_SEGMENT_FILE)
                # Clean up column names
                client_segments = client_segments.rename(columns={
                    'Account Name': 'Client',
                    'Client Segment Type': 'Segment'
                })
                
                # Log Segment distribution before merging
                logger.info(f"Segment distribution before merging: {client_segments['Segment'].value_counts().to_dict()}")

                # Clean up Client names for better matching
                client_segments['Client'] = client_segments['Client'].str.strip().str.upper()
                won_df['Client'] = won_df['Client'].str.strip().str.upper()
                lost_df['Client'] = lost_df['Client'].str.strip().str.upper()

                
                # Merge segments with won and lost dataframes
                won_df = pd.merge(
                    won_df, 
                    client_segments[['Client', 'Segment']], 
                    on='Client', 
                    how='left'
                )
                lost_df = pd.merge(
                    lost_df, 
                    client_segments[['Client', 'Segment']], 
                    on='Client', 
                    how='left'
                )
                
                # Log merge success rate
                won_match = won_df['Segment'].notna().mean() * 100
                lost_match = lost_df['Segment'].notna().mean() * 100
                logger.info(f"Won deals segment match rate: {won_match:.2f}%")
                logger.info(f"Lost deals segment match rate: {lost_match:.2f}%")
                logger.info("Successfully merged client segment data")
            except Exception as e:
                logger.warning(f"Failed to load or merge segment data: {e}")
        
        # Convert CPI columns to numeric before filtering
        won_df['CPI'] = pd.to_numeric(won_df['CPI'], errors='coerce')
        lost_df['CPI'] = pd.to_numeric(lost_df['CPI'], errors='coerce')
        
        # Make sure numeric columns are numeric
        for col in ['IR', 'LOI', 'Completes', 'Revenue']:
            won_df[col] = pd.to_numeric(won_df[col], errors='coerce')
            lost_df[col] = pd.to_numeric(lost_df[col], errors='coerce')
        
        # Filter out invalid CPI values
        won_df = won_df[won_df['CPI'].notna() & (won_df['CPI'] > 0)]
        lost_df = lost_df[lost_df['CPI'].notna() & (lost_df['CPI'] > 0)]
        
        # Log data shapes
        logger.info(f"Won deals count: {len(won_df)}")
        logger.info(f"Lost deals count: {len(lost_df)}")
        
        # Filter out extreme values (over 95th percentile)
        won_percentile_95 = won_df['CPI'].quantile(0.95)
        lost_percentile_95 = lost_df['CPI'].quantile(0.95)
        
        won_df_filtered = won_df[won_df['CPI'] <= won_percentile_95]
        lost_df_filtered = lost_df[lost_df['CPI'] <= lost_percentile_95]
        
        # Determine common columns for combined dataset
        common_columns = ['Client', 'CPI', 'IR', 'LOI', 'Completes', 'Revenue', 'Country', 'Type']
        
        # Add Segment if it exists
        if 'Segment' in won_df.columns and 'Segment' in lost_df.columns:
            common_columns.append('Segment')
        
        # Create a single combined dataframe with only the common columns
        combined_df = pd.concat(
            [won_df[common_columns], lost_df[common_columns]],
            ignore_index=True
        )
        
        # Create a filtered combined dataset
        combined_df_filtered = pd.concat(
            [won_df_filtered[common_columns], lost_df_filtered[common_columns]],
            ignore_index=True
        )
        
        # Return all datasets
        return {
            'won': won_df,
            'won_filtered': won_df_filtered,
            'lost': lost_df,
            'lost_filtered': lost_df_filtered,
            'combined': combined_df,
            'combined_filtered': combined_df_filtered
        }
    
    except Exception as e:
        logger.error(f"Error in load_data: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # Test the data loading function
    try:
        data = load_data()
        print("Data loaded successfully.")
        for key, df in data.items():
            print(f"{key}: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
