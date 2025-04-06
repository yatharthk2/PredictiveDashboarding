"""
ML model training and evaluation for the CPI Analysis & Prediction Dashboard.
Includes functions for building, training, and evaluating prediction models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import joblib
import logging
import os
from typing import Dict, List, Tuple, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Define model configurations
MODEL_CONFIGS = {
    'Linear Regression': {
        'model': LinearRegression(),
        'params': {
            'fit_intercept': [True, False],
            'positive': [True, False]
        }
    },
    'Ridge Regression': {
        'model': Ridge(),
        'params': {
            'alpha': [0.01, 0.1, 1.0, 10.0],
            'fit_intercept': [True, False]
        }
    },
    'Lasso Regression': {
        'model': Lasso(),
        'params': {
            'alpha': [0.01, 0.1, 1.0, 10.0],
            'fit_intercept': [True, False]
        }
    },
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    }
}

def build_models(X: pd.DataFrame, y: pd.Series, 
                do_hyperparameter_tuning: bool = False) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]], pd.DataFrame]:
    """
    Build prediction models for CPI.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        do_hyperparameter_tuning (bool, optional): Whether to perform hyperparameter tuning. Defaults to False.
    
    Returns:
        Tuple[Dict[str, Any], Dict[str, Dict[str, float]], pd.DataFrame]: 
            - Dictionary of trained models
            - Dictionary of model scores
            - DataFrame with feature importance
    """
    try:
        logger.info("Starting model building process")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
        
        # Build models
        if do_hyperparameter_tuning:
            logger.info("Performing hyperparameter tuning")
            models = build_models_with_tuning(X_train, y_train, X_test, y_test)
        else:
            logger.info("Using default model parameters")
            models = build_models_default(X_train, y_train, X_test, y_test)
        
        # Get trained models
        trained_models = models['trained_models']
        model_scores = models['model_scores']
        
        # Get feature importance from Random Forest
        if 'Random Forest' in trained_models:
            rf_model = trained_models['Random Forest']
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            logger.info("Feature importance calculated successfully")
        else:
            # If Random Forest is not available, create empty DataFrame
            feature_importance = pd.DataFrame(columns=['Feature', 'Importance'])
            logger.warning("Could not calculate feature importance: Random Forest model not available")
        
        return trained_models, model_scores, feature_importance
    
    except Exception as e:
        logger.error(f"Error in build_models: {e}", exc_info=True)
        # Return empty objects
        return {}, {}, pd.DataFrame(columns=['Feature', 'Importance'])

def build_models_default(X_train: pd.DataFrame, y_train: pd.Series, 
                        X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Build models with default parameters.
    
    Args:
        X_train (pd.DataFrame): Training feature matrix
        y_train (pd.Series): Training target variable
        X_test (pd.DataFrame): Test feature matrix
        y_test (pd.Series): Test target variable
    
    Returns:
        Dict[str, Any]: Dictionary containing trained models and model scores
    """
    # Initialize dictionaries
    trained_models = {}
    model_scores = {}
    
    # Basic models to train
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    # Train models
    for name, model in models.items():
        logger.info(f"Training {name} model")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        model_scores[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        }
        
        logger.info(f"{name} model trained. R² score: {r2:.4f}")
    
    return {
        'trained_models': trained_models,
        'model_scores': model_scores
    }

def build_models_with_tuning(X_train: pd.DataFrame, y_train: pd.Series, 
                            X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Build models with hyperparameter tuning using GridSearchCV.
    
    Args:
        X_train (pd.DataFrame): Training feature matrix
        y_train (pd.Series): Training target variable
        X_test (pd.DataFrame): Test feature matrix
        y_test (pd.Series): Test target variable
    
    Returns:
        Dict[str, Any]: Dictionary containing tuned models and model scores
    """
    # Initialize dictionaries
    trained_models = {}
    model_scores = {}
    best_params = {}
    
    # Train models with hyperparameter tuning
    for name, config in MODEL_CONFIGS.items():
        logger.info(f"Tuning {name} model")
        
        # Create grid search
        grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Get best model
        trained_models[name] = grid_search.best_estimator_
        best_params[name] = grid_search.best_params_
        
        # Make predictions
        y_pred = grid_search.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        model_scores[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        }
        
        logger.info(f"{name} model tuned. Best params: {grid_search.best_params_}. R² score: {r2:.4f}")
    
    return {
        'trained_models': trained_models,
        'model_scores': model_scores,
        'best_params': best_params
    }

def save_models(models: Dict[str, Any], filename_prefix: str = 'cpi_model') -> Dict[str, str]:
    """
    Save trained models to disk.
    
    Args:
        models (Dict[str, Any]): Dictionary of trained models
        filename_prefix (str, optional): Prefix for filename. Defaults to 'cpi_model'.
    
    Returns:
        Dict[str, str]: Dictionary mapping model names to saved file paths
    """
    try:
        saved_paths = {}
        
        for name, model in models.items():
            # Create a safe filename
            safe_name = name.lower().replace(' ', '_')
            filename = f"{filename_prefix}_{safe_name}.joblib"
            filepath = os.path.join(MODEL_DIR, filename)
            
            # Save model
            joblib.dump(model, filepath)
            saved_paths[name] = filepath
            
            logger.info(f"Saved {name} model to {filepath}")
        
        return saved_paths
    
    except Exception as e:
        logger.error(f"Error saving models: {e}", exc_info=True)
        return {}

def load_models(model_paths: Dict[str, str]) -> Dict[str, Any]:
    """
    Load trained models from disk.
    
    Args:
        model_paths (Dict[str, str]): Dictionary mapping model names to file paths
    
    Returns:
        Dict[str, Any]: Dictionary of loaded models
    """
    try:
        loaded_models = {}
        
        for name, path in model_paths.items():
            # Check if file exists
            if os.path.exists(path):
                # Load model
                model = joblib.load(path)
                loaded_models[name] = model
                
                logger.info(f"Loaded {name} model from {path}")
            else:
                logger.warning(f"Model file not found: {path}")
        
        return loaded_models
    
    except Exception as e:
        logger.error(f"Error loading models: {e}", exc_info=True)
        return {}

def cross_validate_models(X: pd.DataFrame, y: pd.Series, 
                        cv: int = 5) -> Dict[str, Dict[str, float]]:
    """
    Perform cross-validation on models.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        cv (int, optional): Number of cross-validation folds. Defaults to 5.
    
    Returns:
        Dict[str, Dict[str, float]]: Dictionary of cross-validation scores
    """
    try:
        logger.info(f"Starting cross-validation with {cv} folds")
        
        cv_scores = {}
        
        # Basic models to evaluate
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        # Perform cross-validation
        for name, model in models.items():
            logger.info(f"Cross-validating {name} model")
            
            # Calculate scores using cross-validation
            mse_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
            r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            mae_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
            
            # Calculate mean and standard deviation of scores
            cv_scores[name] = {
                'MSE_mean': mse_scores.mean(),
                'MSE_std': mse_scores.std(),
                'RMSE_mean': np.sqrt(mse_scores.mean()),
                'RMSE_std': np.sqrt(mse_scores.std()),
                'MAE_mean': mae_scores.mean(),
                'MAE_std': mae_scores.std(),
                'R²_mean': r2_scores.mean(),
                'R²_std': r2_scores.std()
            }
            
            logger.info(f"{name} cross-validation complete. Mean R²: {r2_scores.mean():.4f}")
        
        return cv_scores
    
    except Exception as e:
        logger.error(f"Error in cross_validate_models: {e}", exc_info=True)
        return {}

if __name__ == "__main__":
    # Test model training with sample data
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            'IR': np.random.uniform(10, 90, n_samples),
            'LOI': np.random.uniform(5, 30, n_samples),
            'Completes': np.random.uniform(100, 1000, n_samples),
            'Type_Won': np.random.binomial(1, 0.5, n_samples)
        })
        
        # Calculate engineered features
        X['IR_LOI_Ratio'] = X['IR'] / X['LOI']
        X['IR_Completes_Ratio'] = X['IR'] / X['Completes']
        
        # Create target variable with some noise
        y = 5 + 0.2 * X['IR'] + 0.5 * X['LOI'] - 0.001 * X['Completes'] + np.random.normal(0, 1, n_samples)
        
        # Test basic model building
        print("Testing basic model building...")
        trained_models, model_scores, feature_importance = build_models(X, y)
        
        # Print results
        print("\nModel scores:")
        for name, scores in model_scores.items():
            print(f"  {name}:")
            for metric, value in scores.items():
                print(f"    {metric}: {value:.4f}")
        
        print("\nFeature importance:")
        print(feature_importance)
        
        # Test model saving and loading
        print("\nTesting model saving and loading...")
        saved_paths = save_models(trained_models)
        loaded_models = load_models(saved_paths)
        
        # Test cross-validation
        print("\nTesting cross-validation...")
        cv_scores = cross_validate_models(X, y, cv=3)
        
        print("\nCross-validation scores:")
        for name, scores in cv_scores.items():
            print(f"  {name}:")
            for metric, value in scores.items():
                print(f"    {metric}: {value:.4f}")
        
        print("\nAll tests completed successfully")
        
    except Exception as e:
        print(f"Error testing models: {e}")
