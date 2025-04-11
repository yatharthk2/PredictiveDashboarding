"""
Data loading functionality for the CPI Analysis & Prediction Dashboard.
Handles loading data from Excel files and initial preprocessing.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, List, Optional, Union
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
        
        # Clean up column values - handle various data issues
        for column in invoiced_df.columns:
            # For string columns, replace null values with empty strings and strip whitespace
            if invoiced_df[column].dtype == 'object':
                invoiced_df[column] = invoiced_df[column].fillna('').astype(str).str.strip()
                
        # Rename columns to remove spaces
        rename_mapping = {
            ' CPI ': 'CPI',
            ' Actual Project Revenue ': 'Revenue',
            'Actual Project Revenue': 'Revenue',
            ' Revenue ': 'Revenue',
            'Revenue ': 'Revenue',
            'Vendor Cost ($)': 'Vendor_Cost',
            ' Vendor Cost ': 'Vendor_Cost'
        }
        
        invoiced_df = invoiced_df.rename(columns={k: v for k, v in rename_mapping.items() if k in invoiced_df.columns})
        
        # Process Countries column
        if 'Countries' in invoiced_df.columns:
            invoiced_df['Countries'] = invoiced_df['Countries'].fillna('[]')
            
            # Extract first country from JSON array format if possible
            def extract_first_country(countries_str):
                countries_str = str(countries_str)  # Ensure it's a string
                try:
                    # Handle basic JSON-like format, e.g. ["United States", "Canada"]
                    if '[' in countries_str and ']' in countries_str:
                        # Extract content between brackets
                        content = countries_str[countries_str.find('[')+1:countries_str.find(']')]
                        if content.strip() == '':
                            return 'USA'  # Default for empty array
                        
                        # Split by comma and handle quotes
                        countries = [c.strip().strip('"\'') for c in content.split(',')]
                        return countries[0] if countries else 'USA'
                    else:
                        return countries_str if countries_str else 'USA'
                except Exception:
                    return countries_str if countries_str else 'USA'
            
            invoiced_df['Country'] = invoiced_df['Countries'].apply(extract_first_country)
            invoiced_df['Country'] = invoiced_df['Country'].replace('', 'USA')
        else:
            # If no Countries column, default to USA
            invoiced_df['Country'] = 'USA'
        
        # Process numeric columns with proper data cleaning
        numeric_columns = {
            'CPI': float,
            'Actual Ir': float,
            'Actual Loi': float,
            'Complete': int,
            'Revenue': float,
            'Project Code Parent': int
        }
        
        for col, dtype in numeric_columns.items():
            if col in invoiced_df.columns:
                try:
                    # For numeric columns, convert to appropriate type safely
                    invoiced_df[col] = pd.to_numeric(
                        invoiced_df[col].astype(str).str.replace('[^0-9.-]', '', regex=True), 
                        errors='coerce'
                    ).astype(dtype)
                    
                    # Log how many values needed to be coerced
                    na_count = invoiced_df[col].isna().sum()
                    if na_count > 0:
                        logger.warning(f"Column {col}: {na_count} values could not be converted to {dtype.__name__}")
                        
                        # Fill missing numeric values with appropriate defaults
                        if col in ['CPI', 'Actual Ir', 'Actual Loi', 'Revenue']:
                            # Use median for continuous variables
                            invoiced_df[col] = invoiced_df[col].fillna(invoiced_df[col].median())
                        elif col in ['Complete', 'Project Code Parent']:
                            # Use 0 for integer counts
                            invoiced_df[col] = invoiced_df[col].fillna(0).astype(int)
                except Exception as e:
                    logger.error(f"Error processing column {col}: {e}")
        
        # Create Won dataset with selected columns
        required_columns = [
            'Project Code Parent', 'Client Name', 'CPI', 'Actual Ir', 'Actual Loi', 
            'Complete', 'Revenue', 'Invoiced Date', 'Country', 'Audience'
        ]
        
        # Create a list of columns that are actually available
        available_columns = [col for col in required_columns if col in invoiced_df.columns]
        
        # Add any missing columns with default values
        for col in required_columns:
            if col not in invoiced_df.columns:
                if col in ['CPI', 'Actual Ir', 'Actual Loi', 'Revenue']:
                    invoiced_df[col] = np.nan
                elif col in ['Complete']:
                    invoiced_df[col] = 0
                else:
                    invoiced_df[col] = ''
                available_columns.append(col)
        
        won_df = invoiced_df[available_columns].copy()
        
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
        
        # Clean up column values for lost deals
        for column in lost_df_raw.columns:
            # For string columns, replace null values with empty strings and strip whitespace
            if lost_df_raw[column].dtype == 'object':
                lost_df_raw[column] = lost_df_raw[column].fillna('').astype(str).str.strip()
        
        # Filter for Sample items only
        lost_df = lost_df_raw[lost_df_raw['Item'] == 'Sample'].copy()
        
        # If no records match, use all data with a warning
        if len(lost_df) == 0:
            logger.warning("No records found with Item='Sample'. Using all lost deal records instead.")
            lost_df = lost_df_raw.copy()
        
        # Create Lost dataset with selected columns
        lost_required_cols = [
            'Record Id', 'Account Name', 'Customer Rate', 'IR', 'LOI', 
            'Qty', 'Item Amount', 'Description (Items)', 'Deal Name'
        ]
        
        # Create a list of columns that are actually available
        lost_available_cols = [col for col in lost_required_cols if col in lost_df.columns]
        
        # Add any missing columns with default values
        for col in lost_required_cols:
            if col not in lost_df.columns:
                if col in ['Customer Rate', 'IR', 'LOI', 'Item Amount']:
                    lost_df[col] = np.nan
                elif col in ['Qty']:
                    lost_df[col] = 0
                else:
                    lost_df[col] = ''
                lost_available_cols.append(col)
        
        lost_df = lost_df[lost_available_cols].copy()
        
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
        
        # Process numeric columns in lost_df
        lost_numeric_cols = {
            'CPI': float,
            'IR': float,
            'LOI': float,
            'Completes': int,
            'Revenue': float
        }
        
        for col, dtype in lost_numeric_cols.items():
            if col in lost_df.columns:
                try:
                    # For numeric columns, convert to appropriate type safely
                    lost_df[col] = pd.to_numeric(
                        lost_df[col].astype(str).str.replace('[^0-9.-]', '', regex=True), 
                        errors='coerce'
                    ).astype(dtype)
                    
                    # Log how many values needed to be coerced
                    na_count = lost_df[col].isna().sum()
                    if na_count > 0:
                        logger.warning(f"Column {col} in lost_df: {na_count} values could not be converted to {dtype.__name__}")
                        
                        # Fill missing numeric values with appropriate defaults
                        if col in ['CPI', 'IR', 'LOI', 'Revenue']:
                            # Use median for continuous variables
                            lost_df[col] = lost_df[col].fillna(lost_df[col].median())
                        elif col in ['Completes']:
                            # Use 0 for integer counts
                            lost_df[col] = lost_df[col].fillna(0).astype(int)
                except Exception as e:
                    logger.error(f"Error processing column {col} in lost_df: {e}")
        
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

                # Merge segments with won and lost dataframes using a fuzzy match strategy
                # First try exact match
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
                
                # Then try with alternative matching for remaining unmatched
                unmatched_won = won_df[won_df['Segment'].isna()]
                unmatched_lost = lost_df[lost_df['Segment'].isna()]
                
                if len(unmatched_won) > 0:
                    # Try substring matching for unmatched clients
                    for idx, row in unmatched_won.iterrows():
                        client_name = row['Client']
                        # Find segments where client name contains the segment client or vice versa
                        matches = client_segments[
                            client_segments['Client'].str.contains(client_name, case=False) |
                            client_name.str.contains(client_segments['Client'], case=False)
                        ]
                        
                        if len(matches) > 0:
                            # Take the first match
                            won_df.loc[idx, 'Segment'] = matches.iloc[0]['Segment']
                
                if len(unmatched_lost) > 0:
                    # Try substring matching for unmatched clients
                    for idx, row in unmatched_lost.iterrows():
                        client_name = row['Client']
                        # Find segments where client name contains the segment client or vice versa
                        matches = client_segments[
                            client_segments['Client'].str.contains(client_name, case=False) |
                            client_name.str.contains(client_segments['Client'], case=False)
                        ]
                        
                        if len(matches) > 0:
                            # Take the first match
                            lost_df.loc[idx, 'Segment'] = matches.iloc[0]['Segment']
                
                # Fill remaining NaN segments with 'Unknown'
                won_df['Segment'] = won_df['Segment'].fillna('Unknown')
                lost_df['Segment'] = lost_df['Segment'].fillna('Unknown')
                
                # Log merge success rate
                won_matched = (won_df['Segment'] != 'Unknown').sum()
                lost_matched = (lost_df['Segment'] != 'Unknown').sum()
                won_match_rate = (won_matched / len(won_df)) * 100
                lost_match_rate = (lost_matched / len(lost_df)) * 100
                
                logger.info(f"Won deals segment match rate: {won_match_rate:.2f}% ({won_matched}/{len(won_df)})")
                logger.info(f"Lost deals segment match rate: {lost_match_rate:.2f}% ({lost_matched}/{len(lost_df)})")
                logger.info("Successfully merged client segment data")
                
            except Exception as e:
                logger.warning(f"Failed to load or merge segment data: {e}")
                # Add default Segment column
                won_df['Segment'] = 'Unknown'
                lost_df['Segment'] = 'Unknown'
        else:
            # Add default Segment column
            won_df['Segment'] = 'Unknown'
            lost_df['Segment'] = 'Unknown'
        
        # Handle extreme outliers in the CPI column
        # Function to cap extreme values
        def cap_outliers(series, quantile=0.95):
            """Cap values above a certain quantile to that quantile's value"""
            cap_value = series.quantile(quantile)
            return series.clip(upper=cap_value)
        
        # Apply outlier capping to numeric columns
        for df in [won_df, lost_df]:
            for col in ['CPI', 'IR', 'LOI', 'Completes', 'Revenue']:
                if col in df.columns:
                    # Cap extreme values
                    df[col] = cap_outliers(df[col], 0.95)
        
        # Ensure all values are valid
        won_df = won_df[won_df['CPI'].notna() & (won_df['CPI'] > 0)]
        lost_df = lost_df[lost_df['CPI'].notna() & (lost_df['CPI'] > 0)]
        
        # Handle zero or near-zero values that could cause mathematical issues
        epsilon = 0.001  # Small positive value to replace zeros
        for df in [won_df, lost_df]:
            for col in ['IR', 'LOI', 'Completes']:
                if col in df.columns:
                    # Replace zeros with a small positive value
                    df.loc[df[col] <= 0, col] = epsilon
        
        # Log data shapes
        logger.info(f"Won deals count: {len(won_df)}")
        logger.info(f"Lost deals count: {len(lost_df)}")
        
        # Create filtered datasets excluding extreme values (over 95th percentile)
        won_percentile_95 = won_df['CPI'].quantile(0.95)
        lost_percentile_95 = lost_df['CPI'].quantile(0.95)
        
        won_df_filtered = won_df[won_df['CPI'] <= won_percentile_95]
        lost_df_filtered = lost_df[lost_df['CPI'] <= lost_percentile_95]
        
        # Add a dataset field to each dataframe to identify its source
        won_df['Dataset'] = 'Won'
        won_df_filtered['Dataset'] = 'Won_Filtered'
        lost_df['Dataset'] = 'Lost'
        lost_df_filtered['Dataset'] = 'Lost_Filtered'
        
        # Log data quality issues
        log_data_quality_issues(won_df, 'Won')
        log_data_quality_issues(lost_df, 'Lost')
        
        # Determine common columns for combined dataset
        common_columns = ['Client', 'CPI', 'IR', 'LOI', 'Completes', 'Revenue', 'Country', 'Type', 'Segment']
        
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

def log_data_quality_issues(df: pd.DataFrame, dataset_name: str) -> None:
    """
    Log potential data quality issues for debugging.
    
    Args:
        df (pd.DataFrame): DataFrame to check
        dataset_name (str): Name of the dataset for logging
    """
    try:
        # Check for missing values
        missing_values = df.isnull().sum()
        missing_cols = missing_values[missing_values > 0]
        if len(missing_cols) > 0:
            logger.warning(f"{dataset_name} dataset has missing values:")
            for col, count in missing_cols.items():
                logger.warning(f"  - {col}: {count} missing values ({count/len(df):.2%})")
        
        # Check for extreme values in numeric columns
        for col in ['CPI', 'IR', 'LOI', 'Completes', 'Revenue']:
            if col in df.columns and df[col].dtype.kind in 'fc':  # Float or complex
                q99 = df[col].quantile(0.99)
                extreme_count = (df[col] > q99).sum()
                if extreme_count > 0:
                    logger.warning(f"{dataset_name} dataset has {extreme_count} values above 99th percentile in {col}")
                    logger.warning(f"  - 99th percentile: {q99:.2f}, max value: {df[col].max():.2f}")
        
        # Check for zero or near-zero values that might cause issues
        for col in ['IR', 'LOI', 'Completes']:
            if col in df.columns:
                zero_count = (df[col] <= 0).sum()
                if zero_count > 0:
                    logger.warning(f"{dataset_name} dataset has {zero_count} zero or negative values in {col}")
        
        # Check for duplicate records
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            logger.warning(f"{dataset_name} dataset has {duplicate_count} duplicate records")
        
        # Check for unusual CPI values (negative or extremely large)
        if 'CPI' in df.columns:
            negative_cpi = (df['CPI'] < 0).sum()
            if negative_cpi > 0:
                logger.warning(f"{dataset_name} dataset has {negative_cpi} negative CPI values")
            
            # Check for suspiciously large CPI values
            large_cpi_threshold = 1000  # Arbitrary threshold for suspicious CPI
            large_cpi = (df['CPI'] > large_cpi_threshold).sum()
            if large_cpi > 0:
                logger.warning(f"{dataset_name} dataset has {large_cpi} CPI values > ${large_cpi_threshold}")
    
    except Exception as e:
        logger.error(f"Error in log_data_quality_issues for {dataset_name}: {e}")

if __name__ == "__main__":
    # Test the data loading function
    try:
        data = load_data()
        print("Data loaded successfully.")
        for key, df in data.items():
            print(f"{key}: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        
        # Try to provide more detailed error information
        import traceback
        print("\nDetailed error information:")
        traceback.print_exc()
        
        # Attempt to check individual files
        print("\nFile existence check:")
        for file_path in [INVOICED_JOBS_FILE, LOST_DEALS_FILE, ACCOUNT_SEGMENT_FILE]:
            exists = os.path.exists(file_path)
            print(f"{file_path}: {'Exists' if exists else 'Missing'}")