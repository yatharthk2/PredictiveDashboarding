"""
Data processing functionality for the CPI Analysis & Prediction Dashboard.
Includes functions for binning data and feature engineering.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Union, Any
from sklearn.preprocessing import StandardScaler

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
        
        # Cap extreme values
        ir_upper_limit = 100  # Incidence rate shouldn't exceed 100%
        df.loc[df['IR'] > ir_upper_limit, 'IR'] = ir_upper_limit
        
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
        
        # Handle extreme LOI outliers (cap at 95th percentile)
        loi_cap = df['LOI'].quantile(0.95)
        df.loc[df['LOI'] > loi_cap, 'LOI'] = loi_cap
        
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
        
        # Handle extreme Completes outliers (cap at 95th percentile)
        completes_cap = df['Completes'].quantile(0.95)
        df.loc[df['Completes'] > completes_cap, 'Completes'] = completes_cap
        
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

def handle_outliers(df: pd.DataFrame, columns: List[str], method: str = 'percentile', threshold: float = 0.95) -> pd.DataFrame:
    """
    Handle outliers in specified columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (List[str]): List of columns to process
        method (str): Method to handle outliers ('percentile', 'zscore', or 'iqr')
        threshold (float): Threshold for outlier detection (0.95 for 95th percentile)
    
    Returns:
        pd.DataFrame: Dataframe with outliers handled
    """
    try:
        df = df.copy()  # Create a copy to avoid modifying the original
        
        for col in columns:
            # Skip if column doesn't exist
            if col not in df.columns:
                continue
                
            # Make sure column is numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Skip columns with all NaN values
            if df[col].isna().all():
                continue
                
            if method == 'percentile':
                # Cap at specified percentile
                cap_value = df[col].quantile(threshold)
                df.loc[df[col] > cap_value, col] = cap_value
                
            elif method == 'zscore':
                # Cap based on z-score
                mean = df[col].mean()
                std = df[col].std()
                z_threshold = 3.0  # Values with z-score > 3 are considered outliers
                df.loc[abs(df[col] - mean) > z_threshold * std, col] = np.sign(df[col] - mean) * z_threshold * std + mean
                
            elif method == 'iqr':
                # Cap based on IQR method
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                upper_bound = q3 + 1.5 * iqr
                df.loc[df[col] > upper_bound, col] = upper_bound
        
        return df
        
    except Exception as e:
        logger.error(f"Error in handle_outliers: {e}", exc_info=True)
        # Return original dataframe if handling fails
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
        
        # First handle outliers in key columns
        key_columns = ['IR', 'LOI', 'Completes', 'CPI']
        df = handle_outliers(df, key_columns, method='percentile', threshold=0.95)
        
        # Make sure key columns are numeric
        for col in key_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with missing values in key columns
        df = df.dropna(subset=key_columns)
        
        # Replace zeros with small values to avoid division by zero
        for col in ['IR', 'LOI', 'Completes']:
            df.loc[df[col] == 0, col] = 0.001
        
        # Basic features
        df['IR_LOI_Ratio'] = df['IR'] / df['LOI']
        df['IR_Completes_Ratio'] = df['IR'] / df['Completes']
        df['LOI_Completes_Ratio'] = df['LOI'] / df['Completes']
        
        # Advanced features
        df['IR_LOI_Product'] = df['IR'] * df['LOI']  # Interaction term
        df['CPI_per_Minute'] = df['CPI'] / df['LOI']  # Cost per minute
        
        # Log transformations for skewed distributions
        df['Log_CPI'] = np.log1p(df['CPI'])  # log(1+x) to handle zeros
        df['Log_Completes'] = np.log1p(df['Completes'])
        df['Log_IR_LOI_Product'] = np.log1p(df['IR_LOI_Product'])
        
        # Scaling might be useful for some visualization techniques
        # But we don't apply it here as it changes units interpretability
        
        # Efficiency metric
        df['CPI_Efficiency'] = (df['IR'] / 100) * (1 / df['LOI']) * np.sqrt(df['Completes'])
        
        # Replace any infinite values that might have been created during division
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill any new NaN values with median
        for col in df.columns:
            if df[col].dtype.kind in 'fc':  # Only numeric columns
                if df[col].isna().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
        
        return df
    
    except Exception as e:
        logger.error(f"Error in engineer_features: {e}", exc_info=True)
        # Return original dataframe if feature engineering fails
        return df

def prepare_model_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for modeling with improved error handling tailored to CPI data.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Feature matrix X and target vector y
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
        
        # Handle outliers in key columns
        df = handle_outliers(df, ['IR', 'LOI', 'Completes', 'CPI'], method='percentile', threshold=0.95)
        
        # Handle missing values
        numeric_cols = ['IR', 'LOI', 'Completes', 'CPI']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            na_count = df[col].isna().sum()
            if na_count > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.warning(f"Column {col}: Replaced {na_count} NaN values with median {median_val:.2f}")
            if col in ['LOI', 'Completes'] and (df[col] <= 0).any():
                min_non_zero = max(0.1, df[df[col] > 0][col].min())
                zero_count = (df[col] <= 0).sum()
                df.loc[df[col] <= 0, col] = min_non_zero
                logger.warning(f"Column {col}: Replaced {zero_count} zero/negative values with {min_non_zero}")
        
        # Create engineered features specific to CPI analysis
        try:
            # Basic ratios
            df['IR_LOI_Ratio'] = df['IR'] / df['LOI']
            df['IR_Completes_Ratio'] = df['IR'] / df['Completes']
            df['LOI_Completes_Ratio'] = df['LOI'] / df['Completes']
            
            # Interaction terms
            df['IR_LOI_Product'] = df['IR'] * df['LOI']
            
            # Log transformations for skewed variables
            df['Log_CPI'] = np.log1p(df['CPI'])
            df['Log_Completes'] = np.log1p(df['Completes'])
            df['Log_IR_LOI_Product'] = np.log1p(df['IR_LOI_Product'])
            
            # Polynomial features for non-linear relationships
            df['IR_Squared'] = df['IR'] ** 2
            df['LOI_Squared'] = df['LOI'] ** 2
            
            # Create bins if not already present
            if 'IR_Bin' not in df.columns:
                df = create_ir_bins(df)
                
            logger.info("Successfully created engineered features")
        except Exception as e:
            logger.warning(f"Error in feature engineering: {e}")
        
        # Remove any infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill any remaining NaN values with median for each column
        for col in df.columns:
            if df[col].dtype.kind in 'fc':  # Only numeric columns
                if df[col].isna().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
        
        # Define feature columns, excluding the target variable
        feature_cols = [
            'IR', 'LOI', 'Completes', 'IR_LOI_Ratio', 
            'IR_Completes_Ratio', 'Log_Completes', 'IR_LOI_Product',
            'Log_IR_LOI_Product', 'IR_Squared', 'LOI_Squared',
            'Type'
        ]
        
        # Keep only available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        
        # Handle categorical Type column
        if 'Type' in available_cols:
            valid_types = ['Won', 'Lost']
            invalid_types = df[~df['Type'].isin(valid_types)]['Type'].unique()
            if len(invalid_types) > 0:
                logger.warning(f"Found invalid Type values: {invalid_types}. Removing these rows.")
                df = df[df['Type'].isin(valid_types)]
            
            # One-hot encode the Type column
            df = pd.get_dummies(df, columns=['Type'], drop_first=True)
            if 'Type_Won' in df.columns:
                available_cols = [col for col in available_cols if col != 'Type'] + ['Type_Won']
        
        # Add segment information if available
        if 'Segment' in df.columns and df['Segment'].notna().any():
            logger.info("Client segment data found, adding to model features")
            df = pd.get_dummies(df, columns=['Segment'], drop_first=True)
            segment_cols = [col for col in df.columns if col.startswith('Segment_')]
            available_cols.extend(segment_cols)
        
        # Add any additional demographic features if available
        for demo_col in ['Country', 'Audience']:
            if demo_col in df.columns and df[demo_col].notna().any():
                # Limit to top N categories to avoid too many dummy variables
                top_categories = df[demo_col].value_counts().nlargest(5).index
                df[f'{demo_col}_Other'] = ~df[demo_col].isin(top_categories)
                # Only include top categories in dummy variables
                df_filtered = df.copy()
                df_filtered.loc[df_filtered[demo_col].isin(top_categories) == False, demo_col] = 'Other'
                df = pd.get_dummies(df_filtered, columns=[demo_col], drop_first=True)
                demo_cols = [col for col in df.columns if col.startswith(f'{demo_col}_')]
                available_cols.extend(demo_cols)
        
        # Prepare final feature matrix and target vector
        X = df[available_cols].copy()
        y = df['CPI']
        
        # Make sure all values are valid
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]
        
        # Check if we have enough data for modeling
        if len(X) < 10:
            logger.error(f"Too few samples after preprocessing: {len(X)}")
            return pd.DataFrame(), pd.Series()
        
        # Scale the features to improve SVD convergence
        numeric_features = X.select_dtypes(include=['float', 'int']).columns
        if len(numeric_features) > 0:
            scaler = StandardScaler()
            X[numeric_features] = scaler.fit_transform(X[numeric_features])
            logger.info("Features scaled to improve numerical stability")
        
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
        # Handle outliers before calculating summary statistics
        df_clean = handle_outliers(df, ['CPI', 'IR', 'LOI', 'Completes'], method='percentile', threshold=0.95)
        
        # Group by Type
        grouped = df_clean.groupby('Type')
        
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
                'CPI_95th': group['CPI'].quantile(0.95),
                'CPI_StdDev': group['CPI'].std()
            }
            
            # Add log-transformed means for skewed variables
            # This gives a better representation of central tendency for skewed data
            summary[name]['Log_Avg_CPI'] = np.exp(np.log1p(group['CPI']).mean())
            summary[name]['Log_Avg_Completes'] = np.exp(np.log1p(group['Completes']).mean())
    
    except Exception as e:
        logger.error(f"Error in get_data_summary: {e}", exc_info=True)
    
    return summary

def detect_data_issues(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect potential issues in the data that could cause modeling problems.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        Dict[str, Any]: Dictionary of detected issues
    """
    issues = {
        'extreme_outliers': {},
        'missing_values': {},
        'zeros': {},
        'skewness': {},
        'collinearity': {}
    }
    
    try:
        # Check for extreme outliers (beyond 10 standard deviations)
        for col in ['CPI', 'IR', 'LOI', 'Completes']:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                threshold = mean + 10 * std
                extreme_count = (df[col] > threshold).sum()
                if extreme_count > 0:
                    issues['extreme_outliers'][col] = {
                        'count': int(extreme_count),
                        'percentage': float((extreme_count / len(df)) * 100),
                        'threshold': float(threshold),
                        'max_value': float(df[col].max())
                    }
        
        # Check for missing values
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                issues['missing_values'][col] = {
                    'count': int(missing_count),
                    'percentage': float((missing_count / len(df)) * 100)
                }
        
        # Check for zeros in columns where zeros are problematic
        for col in ['LOI', 'Completes']:
            if col in df.columns:
                zero_count = (df[col] == 0).sum()
                if zero_count > 0:
                    issues['zeros'][col] = {
                        'count': int(zero_count),
                        'percentage': float((zero_count / len(df)) * 100)
                    }
        
        # Check for skewness
        for col in ['CPI', 'IR', 'LOI', 'Completes']:
            if col in df.columns:
                mean = df[col].mean()
                median = df[col].median()
                skewness = float(3 * (mean - median) / (df[col].std() if df[col].std() > 0 else 1))
                if abs(skewness) > 1:
                    issues['skewness'][col] = {
                        'skewness': skewness,
                        'mean': float(mean),
                        'median': float(median)
                    }
        
        # Check for collinearity
        numeric_cols = df.select_dtypes(include=['float', 'int']).columns.tolist()
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            high_corr_pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr = corr_matrix.iloc[i, j]
                    if abs(corr) > 0.8:
                        high_corr_pairs.append({
                            'var1': numeric_cols[i],
                            'var2': numeric_cols[j],
                            'correlation': float(corr)
                        })
            
            if high_corr_pairs:
                issues['collinearity']['high_correlation_pairs'] = high_corr_pairs
    
    except Exception as e:
        logger.error(f"Error in detect_data_issues: {e}", exc_info=True)
        issues['error'] = str(e)
    
    return issues

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
        
        # Test data issue detection
        issues = detect_data_issues(df)
        print("\nDetected Data Issues:")
        print(issues)
        
    except Exception as e:
        print(f"Error testing data processor: {e}")