"""
Data processing functionality for the CPI Analysis & Prediction Dashboard.
Includes functions for binning data and feature engineering.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define bin configurations
IR_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
IR_BIN_LABELS = ['0-10', '10-20', '20-30', '30-40', '40-50', 
                '50-60', '60-70', '70-80', '80-90', '90-100']

LOI_BINS = [0, 5, 10, 15, 20, float('inf')]
LOI_BIN_LABELS = ['Very Short (1-5 min)', 'Short (6-10 min)', 
                 'Medium (11-15 min)', 'Long (16-20 min)', 'Very Long (20+ min)']

COMPLETES_BINS = [0, 100, 500, 1000, float('inf')]
COMPLETES_BIN_LABELS = ['Small (1-100)', 'Medium (101-500)', 
                       'Large (501-1000)', 'Very Large (1000+)']

def create_ir_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create IR (Incidence Rate) bins for analysis.
    
    Args:
        df (pd.DataFrame): Input dataframe with IR column
    
    Returns:
        pd.DataFrame: Dataframe with added IR_Bin column
    """
    try:
        df = df.copy()  # Create a copy to avoid modifying the original
        
        # Make sure IR is numeric
        df['IR'] = pd.to_numeric(df['IR'], errors='coerce')
        
        # Create bins
        df['IR_Bin'] = pd.cut(
            df['IR'],
            bins=IR_BINS,
            labels=IR_BIN_LABELS
        )
        
        return df
    
    except Exception as e:
        logger.error(f"Error in create_ir_bins: {e}", exc_info=True)
        # Return original dataframe if binning fails
        return df

def create_loi_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create LOI (Length of Interview) bins for analysis.
    
    Args:
        df (pd.DataFrame): Input dataframe with LOI column
    
    Returns:
        pd.DataFrame: Dataframe with added LOI_Bin column
    """
    try:
        df = df.copy()  # Create a copy to avoid modifying the original
        
        # Make sure LOI is numeric
        df['LOI'] = pd.to_numeric(df['LOI'], errors='coerce')
        
        # Create bins
        df['LOI_Bin'] = pd.cut(
            df['LOI'],
            bins=LOI_BINS,
            labels=LOI_BIN_LABELS
        )
        
        return df
    
    except Exception as e:
        logger.error(f"Error in create_loi_bins: {e}", exc_info=True)
        # Return original dataframe if binning fails
        return df

def create_completes_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Sample Size (Completes) bins for analysis.
    
    Args:
        df (pd.DataFrame): Input dataframe with Completes column
    
    Returns:
        pd.DataFrame: Dataframe with added Completes_Bin column
    """
    try:
        df = df.copy()  # Create a copy to avoid modifying the original
        
        # Make sure Completes is numeric
        df['Completes'] = pd.to_numeric(df['Completes'], errors='coerce')
        
        # Create bins
        df['Completes_Bin'] = pd.cut(
            df['Completes'],
            bins=COMPLETES_BINS,
            labels=COMPLETES_BIN_LABELS
        )
        
        return df
    
    except Exception as e:
        logger.error(f"Error in create_completes_bins: {e}", exc_info=True)
        # Return original dataframe if binning fails
        return df

def apply_all_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all binning functions to a dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Dataframe with all bin columns added
    """
    df = create_ir_bins(df)
    df = create_loi_bins(df)
    df = create_completes_bins(df)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Dataframe with additional engineered features
    """
    try:
        df = df.copy()  # Create a copy to avoid modifying the original
        
        # Make sure key columns are numeric
        for col in ['IR', 'LOI', 'Completes', 'CPI']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with missing values in key columns
        df = df.dropna(subset=['IR', 'LOI', 'Completes', 'CPI'])
        
        # Basic features
        df['IR_LOI_Ratio'] = df['IR'] / df['LOI']
        df['IR_Completes_Ratio'] = df['IR'] / df['Completes']
        df['LOI_Completes_Ratio'] = df['LOI'] / df['Completes']
        
        # Advanced features
        df['IR_LOI_Product'] = df['IR'] * df['LOI']  # Interaction term
        df['CPI_per_Minute'] = df['CPI'] / df['LOI']  # Cost per minute
        df['Log_Completes'] = np.log1p(df['Completes'])  # Log transformation for skewed distribution
        
        # Efficiency metric
        df['CPI_Efficiency'] = (df['IR'] / 100) * (1 / df['LOI']) * df['Completes']
        
        # Replace any infinite values that might have been created during division
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df
    
    except Exception as e:
        logger.error(f"Error in engineer_features: {e}", exc_info=True)
        # Return original dataframe if feature engineering fails
        return df

def prepare_model_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for modeling with improved error handling tailored to CPI data.
    """
    try:
        logger.info("Starting data preparation for modeling")
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Validate required columns exist
        required_cols = ['IR', 'LOI', 'Completes', 'CPI', 'Type']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {', '.join(missing_cols)}")
            return pd.DataFrame(), pd.Series()
        
        # Handle missing values
        numeric_cols = ['IR', 'LOI', 'Completes', 'CPI']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            na_count = df[col].isna().sum()
            if na_count > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.warning(f"Column {col}: Replaced {na_count} NaN values with median {median_val:.2f}")
            if col in ['LOI', 'Completes'] and (df[col] == 0).any():
                min_non_zero = df[df[col] > 0][col].min()
                df.loc[df[col] == 0, col] = min_non_zero
                logger.warning(f"Column {col}: Replaced zero values with minimum non-zero value {min_non_zero}")
        
        # Create engineered features specific to CPI analysis
        try:
            df['IR_LOI_Ratio'] = df['IR'] / df['LOI']
            df['IR_Completes_Ratio'] = df['IR'] / df['Completes']
            df['LOI_Completes_Ratio'] = df['LOI'] / df['Completes']
            df['IR_LOI_Product'] = df['IR'] * df['LOI']
            df['Log_Completes'] = np.log1p(df['Completes'])
            
            if 'IR_Bin' not in df.columns:
                from utils.data_processor import create_ir_bins
                df = create_ir_bins(df)
                
            logger.info("Successfully created engineered features")
        except Exception as e:
            logger.warning(f"Error in feature engineering: {e}")
        
        feature_cols = [
            'IR', 'LOI', 'Completes', 'IR_LOI_Ratio', 
            'IR_Completes_Ratio', 'Log_Completes', 'Type'
        ]
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if 'Type' in available_cols:
            valid_types = ['Won', 'Lost']
            invalid_types = df[~df['Type'].isin(valid_types)]['Type'].unique()
            if len(invalid_types) > 0:
                logger.warning(f"Found invalid Type values: {invalid_types}. Removing these rows.")
                df = df[df['Type'].isin(valid_types)]
            df = pd.get_dummies(df, columns=['Type'], drop_first=True)
            if 'Type_Won' in df.columns:
                available_cols = [col for col in available_cols if col != 'Type'] + ['Type_Won']
        
        if 'Segment' in df.columns and df['Segment'].notna().any():
            logger.info("Client segment data found, adding to model features")
            df = pd.get_dummies(df, columns=['Segment'], drop_first=True)
            segment_cols = [col for col in df.columns if col.startswith('Segment_')]
            available_cols.extend(segment_cols)
        
        X = df[available_cols].copy()
        y = df['CPI']
        
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]
        
        if len(X) < 10:
            logger.error(f"Too few samples after preprocessing: {len(X)}")
            return pd.DataFrame(), pd.Series()
        
        logger.info(f"Data preparation successful. X shape: {X.shape}, y shape: {y.shape}")
        logger.info(f"Features used: {', '.join(X.columns)}")
        
        return X, y
        
    except Exception as e:
        logger.error(f"Error in prepare_model_data: {e}", exc_info=True)
        return pd.DataFrame(), pd.Series()


def get_data_summary(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Generate summary statistics for key metrics by Type (Won/Lost).
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        Dict[str, Dict[str, float]]: Dictionary of summary metrics by Type
    """
    summary = {}
    
    try:
        # Group by Type
        grouped = df.groupby('Type')
        
        # Calculate summary statistics for each group
        for name, group in grouped:
            summary[name] = {
                'Count': len(group),
                'Avg_CPI': group['CPI'].mean(),
                'Median_CPI': group['CPI'].median(),
                'Avg_IR': group['IR'].mean(),
                'Avg_LOI': group['LOI'].mean(),
                'Avg_Completes': group['Completes'].mean(),
                'CPI_25th': group['CPI'].quantile(0.25),
                'CPI_75th': group['CPI'].quantile(0.75),
                'CPI_95th': group['CPI'].quantile(0.95)
            }
    
    except Exception as e:
        logger.error(f"Error in get_data_summary: {e}", exc_info=True)
    
    return summary

if __name__ == "__main__":
    # Test the data processing functions with a sample dataframe
    try:
        import pandas as pd
        
        # Create a sample dataframe
        data = {
            'CPI': [10.5, 12.3, 8.7, 15.2, 9.8],
            'IR': [25, 35, 45, 15, 55],
            'LOI': [10, 15, 8, 20, 12],
            'Completes': [500, 300, 800, 200, 600],
            'Type': ['Won', 'Lost', 'Won', 'Lost', 'Won']
        }
        
        df = pd.DataFrame(data)
        
        # Test binning
        binned_df = apply_all_bins(df)
        print("Binned DataFrame:")
        print(binned_df.head())
        
        # Test feature engineering
        engineered_df = engineer_features(df)
        print("\nEngineered DataFrame:")
        print(engineered_df.head())
        
        # Test model data preparation
        X, y = prepare_model_data(engineered_df)
        print("\nModel Input X:")
        print(X.head())
        print("\nModel Target y:")
        print(y.head())
        
    except Exception as e:
        print(f"Error testing data processor: {e}")
