"""
ML model training and evaluation for the CPI Analysis & Prediction Dashboard.
Includes functions for building, training, and evaluating prediction models.
"""

import pandas as pd
import numpy as np
import traceback
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor, RANSACRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
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
        'model': Ridge(random_state=42),
        'params': {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
            'fit_intercept': [True, False],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        }
    },
    'Huber Regression': {
        'model': HuberRegressor(),
        'params': {
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'epsilon': [1.1, 1.35, 1.5, 2.0],
            'max_iter': [100, 500, 1000]
        }
    },
    'RANSAC Regression': {
        'model': RANSACRegressor(random_state=42),
        'params': {
            'min_samples': [0.1, 0.5, 0.9],
            'max_trials': [50, 100, 200],
            'loss': ['absolute_loss', 'squared_loss']
        }
    },
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'subsample': [0.8, 0.9, 1.0]
        }
    }
}

def build_models(X: pd.DataFrame, y: pd.Series, 
                do_hyperparameter_tuning: bool = False) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]], pd.DataFrame]:
    """
    Build prediction models for CPI with enhanced robustness and stability.
    
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
        
        # Validate input data
        if X.empty or len(y) == 0:
            logger.error("Empty input data provided")
            return {}, {}, pd.DataFrame(columns=['Feature', 'Importance'])
        
        # Check for NaN or infinite values
        if X.isnull().any().any() or np.isinf(X).any().any() or y.isnull().any() or np.isinf(y).any():
            logger.warning("Input data contains NaN or infinite values - attempting to clean data")
            
            # Replace NaN values with median
            for col in X.columns:
                X[col] = X[col].fillna(X[col].median())
            
            # Replace infinite values with large but finite values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
            
            # Handle target variable
            y = y.fillna(y.median())
            y = y.replace([np.inf, -np.inf], y.median())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
        
        # Check for sufficient data
        if len(X_train) < 10 or len(X_test) < 5:
            logger.warning(f"Very small dataset: {len(X_train)} training samples, {len(X_test)} test samples")
            if len(X_train) < 5:
                logger.error("Insufficient data for modeling")
                return {}, {}, pd.DataFrame(columns=['Feature', 'Importance'])
        
        # Build models with or without hyperparameter tuning
        if do_hyperparameter_tuning:
            logger.info("Performing hyperparameter tuning")
            models = build_models_with_tuning(X_train, y_train, X_test, y_test)
        else:
            logger.info("Using default model parameters")
            models = build_models_default(X_train, y_train, X_test, y_test)
        
        # Get trained models
        trained_models = models.get('trained_models', {})
        model_scores = models.get('model_scores', {})
        
        # Extract feature importance (try from multiple models if available)
        feature_importance = pd.DataFrame(columns=['Feature', 'Importance'])
        
        # Try to get feature importance from multiple model types
        importance_models = ['Random Forest', 'Gradient Boosting']
        for model_name in importance_models:
            if model_name in trained_models:
                try:
                    model = trained_models[model_name]
                    if hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                        feature_importance = pd.DataFrame({
                            'Feature': X.columns,
                            'Importance': importance
                        }).sort_values('Importance', ascending=False)
                        logger.info(f"Feature importance calculated from {model_name}")
                        break
                except Exception as e:
                    logger.warning(f"Could not extract feature importance from {model_name}: {e}")
        
        if feature_importance.empty:
            logger.warning("Could not calculate feature importance from any model")
            # Create empty DataFrame with proper structure
            feature_importance = pd.DataFrame(columns=['Feature', 'Importance'])
        
        return trained_models, model_scores, feature_importance
    
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error in build_models: {error_details}")
        # Return empty objects
        return {}, {}, pd.DataFrame(columns=['Feature', 'Importance'])

def build_models_default(X_train: pd.DataFrame, y_train: pd.Series, 
                        X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Build models with default parameters but added robustness.
    
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
    
    # Define models to train
    models = {
        'Ridge Regression': Ridge(alpha=1.0, random_state=42, solver='lsqr'),
        'Huber Regression': HuberRegressor(alpha=0.01, epsilon=1.35),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    # Use a scaler to avoid numerical issues
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models with error handling for each model
    for name, model in models.items():
        try:
            logger.info(f"Training {name} model")
            model.fit(X_train_scaled, y_train)
            trained_models[name] = model
            
            # Make predictions
            try:
                y_pred = model.predict(X_test_scaled)
                
                # Ensure predictions are within reasonable bounds
                y_pred = np.clip(y_pred, 0, y_test.max() * 3)
                
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
            except Exception as eval_error:
                logger.error(f"Error evaluating {name} model: {eval_error}")
                model_scores[name] = {
                    'MSE': float('nan'),
                    'RMSE': float('nan'),
                    'MAE': float('nan'),
                    'R²': float('nan'),
                    'Error': str(eval_error)
                }
        except Exception as e:
            logger.error(f"Error training {name} model: {e}")
            # Model failed, don't add it to trained_models
            model_scores[name] = {
                'Error': str(e),
                'Status': 'Failed'
            }
    
    # If no models were successfully trained, try a very basic linear model as fallback
    if not trained_models:
        try:
            logger.warning("All standard models failed, attempting basic linear regression as fallback")
            fallback_model = LinearRegression()
            fallback_model.fit(X_train_scaled, y_train)
            trained_models['Fallback Linear'] = fallback_model
            
            # Evaluate fallback model
            y_pred = fallback_model.predict(X_test_scaled)
            y_pred = np.clip(y_pred, 0, y_test.max() * 3)
            
            mse = mean_squared_error(y_test, y_pred)
            model_scores['Fallback Linear'] = {
                'MSE': mse,
                'RMSE': np.sqrt(mse),
                'MAE': mean_absolute_error(y_test, y_pred),
                'R²': r2_score(y_test, y_pred)
            }
        except Exception as e:
            logger.error(f"Even fallback model failed: {e}")
    
    return {
        'trained_models': trained_models,
        'model_scores': model_scores,
        'scaler': scaler  # Include the scaler for future preprocessing
    }

def build_models_with_tuning(X_train: pd.DataFrame, y_train: pd.Series, 
                            X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Build models with hyperparameter tuning using GridSearchCV and added robustness.
    
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
    
    # Use a scaler to ensure numerical stability
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models with hyperparameter tuning - limit to fewer models for efficiency
    tuning_models = ['Ridge Regression', 'Random Forest', 'Gradient Boosting']
    
    for name in tuning_models:
        if name in MODEL_CONFIGS:
            try:
                logger.info(f"Tuning {name} model")
                config = MODEL_CONFIGS[name]
                
                # Create grid search with reduced parameter space if data is small
                if len(X_train) < 50:
                    logger.warning(f"Small dataset detected, reducing parameter space for {name}")
                    # Simplify params for small datasets
                    simplified_params = {}
                    for param, values in config['params'].items():
                        if isinstance(values, list) and len(values) > 2:
                            simplified_params[param] = [values[0], values[-1]]
                        else:
                            simplified_params[param] = values
                    params = simplified_params
                else:
                    params = config['params']
                
                # Create grid search with appropriate CV
                cv = min(5, len(X_train) // 5)  # Reduce CV folds for small datasets
                cv = max(2, cv)  # Ensure at least 2-fold CV
                
                grid_search = GridSearchCV(
                    config['model'],
                    params,
                    cv=cv,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    error_score='raise',
                    verbose=1
                )
                
                # Fit grid search with timeout and error handling
                try:
                    grid_search.fit(X_train_scaled, y_train)
                    
                    # Get best model
                    trained_models[name] = grid_search.best_estimator_
                    best_params[name] = grid_search.best_params_
                    
                    # Make predictions
                    y_pred = grid_search.predict(X_test_scaled)
                    
                    # Ensure predictions are within reasonable bounds
                    y_pred = np.clip(y_pred, 0, y_test.max() * 3)
                    
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
                
                except Exception as grid_error:
                    logger.error(f"Grid search failed for {name}: {grid_error}")
                    
                    # Try with default parameters instead
                    try:
                        logger.info(f"Falling back to default parameters for {name}")
                        model = config['model']
                        model.fit(X_train_scaled, y_train)
                        trained_models[name] = model
                        
                        # Evaluate with default parameters
                        y_pred = model.predict(X_test_scaled)
                        y_pred = np.clip(y_pred, 0, y_test.max() * 3)
                        
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        model_scores[name] = {
                            'MSE': mse,
                            'RMSE': rmse,
                            'MAE': mae,
                            'R²': r2,
                            'Note': 'Using default parameters (GridSearch failed)'
                        }
                        
                        logger.info(f"{name} model trained with defaults. R² score: {r2:.4f}")
                    except Exception as e:
                        logger.error(f"Even default training failed for {name}: {e}")
                        model_scores[name] = {
                            'Error': str(e),
                            'Status': 'Failed'
                        }
            
            except Exception as e:
                error_details = traceback.format_exc()
                logger.error(f"Error in tuning {name} model: {error_details}")
                model_scores[name] = {
                    'Error': str(e),
                    'Status': 'Failed'
                }
    
    # If no models were successfully trained, try a very basic model as fallback
    if not trained_models:
        try:
            logger.warning("All tuned models failed, attempting basic linear regression as fallback")
            fallback_model = LinearRegression()
            fallback_model.fit(X_train_scaled, y_train)
            trained_models['Fallback Linear'] = fallback_model
            
            # Evaluate fallback model
            y_pred = fallback_model.predict(X_test_scaled)
            y_pred = np.clip(y_pred, 0, y_test.max() * 3)
            
            mse = mean_squared_error(y_test, y_pred)
            model_scores['Fallback Linear'] = {
                'MSE': mse,
                'RMSE': np.sqrt(mse),
                'MAE': mean_absolute_error(y_test, y_pred),
                'R²': r2_score(y_test, y_pred)
            }
        except Exception as e:
            logger.error(f"Even fallback model failed: {e}")
    
    return {
        'trained_models': trained_models,
        'model_scores': model_scores,
        'best_params': best_params,
        'scaler': scaler  # Include the scaler for future preprocessing
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
        error_details = traceback.format_exc()
        logger.error(f"Error saving models: {error_details}")
        return {}

def save_model_pipeline(models: Dict[str, Any], scaler: Any, filename_prefix: str = 'cpi_pipeline') -> Dict[str, str]:
    """
    Save complete model pipelines including preprocessing to disk.
    
    Args:
        models (Dict[str, Any]): Dictionary of trained models
        scaler: The fitted scaler used for preprocessing
        filename_prefix (str, optional): Prefix for filename. Defaults to 'cpi_pipeline'.
    
    Returns:
        Dict[str, str]: Dictionary mapping model names to saved pipeline paths
    """
    try:
        saved_paths = {}
        
        for name, model in models.items():
            # Create a pipeline that includes preprocessing
            pipeline = Pipeline([
                ('scaler', scaler),
                ('model', model)
            ])
            
            # Create a safe filename
            safe_name = name.lower().replace(' ', '_')
            filename = f"{filename_prefix}_{safe_name}.joblib"
            filepath = os.path.join(MODEL_DIR, filename)
            
            # Save pipeline
            joblib.dump(pipeline, filepath)
            saved_paths[name] = filepath
            
            logger.info(f"Saved {name} pipeline to {filepath}")
        
        return saved_paths
    
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error saving model pipelines: {error_details}")
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
        error_details = traceback.format_exc()
        logger.error(f"Error loading models: {error_details}")
        return {}

def cross_validate_models(X: pd.DataFrame, y: pd.Series, 
                        cv: int = 5) -> Dict[str, Dict[str, float]]:
    """
    Perform cross-validation on models with added robustness.
    
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
        
        # Adjust CV based on data size
        if len(X) < cv * 2:
            adjusted_cv = max(2, len(X) // 2)
            logger.warning(f"Dataset too small for {cv}-fold CV, adjusting to {adjusted_cv}-fold")
            cv = adjusted_cv
        
        # Handle NaN or infinite values
        X = X.fillna(X.median())
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        y = y.fillna(y.median())
        y = y.replace([np.inf, -np.inf], y.median())
        
        # Apply robust scaling
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Basic models to evaluate - use more robust models
        models = {
            'Ridge Regression': Ridge(alpha=1.0, random_state=42, solver='lsqr'),
            'Huber Regression': HuberRegressor(alpha=0.01, epsilon=1.35),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        # Perform cross-validation
        for name, model in models.items():
            try:
                logger.info(f"Cross-validating {name} model")
                
                # Calculate scores using cross-validation
                mse_scores = -cross_val_score(model, X_scaled, y, cv=cv, scoring='neg_mean_squared_error')
                r2_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')
                mae_scores = -cross_val_score(model, X_scaled, y, cv=cv, scoring='neg_mean_absolute_error')
                
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
            
            except Exception as e:
                logger.error(f"Error in cross-validation for {name}: {e}")
                cv_scores[name] = {
                    'Error': str(e),
                    'Status': 'Failed'
                }
        
        return cv_scores
    
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error in cross_validate_models: {error_details}")
        return {}

def evaluate_model_assumptions(X: pd.DataFrame, y: pd.Series, model: Any) -> Dict[str, Any]:
    """
    Evaluate key assumptions for regression models.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        model: Trained model to evaluate
    
    Returns:
        Dict[str, Any]: Dictionary with assumption test results
    """
    try:
        assumptions = {}
        
        # 1. Check for linearity (for linear models)
        if hasattr(model, 'coef_'):
            # Make predictions
            X_scaled = StandardScaler().fit_transform(X)
            y_pred = model.predict(X_scaled)
            
            # Calculate residuals
            residuals = y - y_pred
            
            # Check residual normality
            from scipy import stats
            _, p_value = stats.normaltest(residuals)
            assumptions['residual_normality'] = {
                'p_value': p_value,
                'normal_distribution': p_value > 0.05
            }
            
            # Check homoscedasticity (constant variance)
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            
            # Create polynomial features for Breusch-Pagan test
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly.fit_transform(X_scaled)
            
            # Fit a model to predict squared residuals
            bp_model = LinearRegression()
            bp_model.fit(X_poly, residuals**2)
            
            # Calculate BP test statistic (simplified)
            n = len(X)
            bp_statistic = n * bp_model.score(X_poly, residuals**2)
            bp_p_value = 1 - stats.chi2.cdf(bp_statistic, X_poly.shape[1])
            
            assumptions['homoscedasticity'] = {
                'test': 'Breusch-Pagan',
                'statistic': bp_statistic,
                'p_value': bp_p_value,
                'constant_variance': bp_p_value > 0.05
            }
        
        # 2. Check for multicollinearity
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        # Calculate VIF for each feature
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        
        # Handle potential singular matrix by adding small constant
        X_vif = X.copy() + 1e-10
        
        try:
            vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
            
            # Check for multicollinearity issue
            high_vif_features = vif_data[vif_data["VIF"] > 10]
            
            assumptions['multicollinearity'] = {
                'vif_values': vif_data.to_dict('records'),
                'high_vif_features': high_vif_features.to_dict('records'),
                'has_multicollinearity': len(high_vif_features) > 0
            }
        except Exception as vif_error:
            logger.warning(f"Could not calculate VIF values: {vif_error}")
            assumptions['multicollinearity'] = {
                'error': str(vif_error),
                'status': 'Failed'
            }
        
        # 3. Check for outliers in the data
        from sklearn.ensemble import IsolationForest
        
        # Use Isolation Forest for outlier detection
        isolation_forest = IsolationForest(random_state=42)
        outliers = isolation_forest.fit_predict(X_scaled)
        
        # Count outliers (predicted as -1)
        outlier_count = (outliers == -1).sum()
        outlier_percent = (outlier_count / len(X)) * 100
        
        assumptions['outliers'] = {
            'count': int(outlier_count),
            'percentage': float(outlier_percent),
            'high_outliers': outlier_percent > 5
        }
        
        return assumptions
    
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error in evaluate_model_assumptions: {error_details}")
        return {'error': str(e), 'status': 'Failed'}

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
        
        # Test model assumption evaluation
        if trained_models:
            print("\nTesting model assumption evaluation...")
            first_model_name = list(trained_models.keys())[0]
            first_model = trained_models[first_model_name]
            
            assumptions = evaluate_model_assumptions(X, y, first_model)
            print(f"Model assumptions for {first_model_name}:")
            import json
            print(json.dumps(assumptions, indent=2))
        
        print("\nAll tests completed successfully")
        
    except Exception as e:
        print(f"Error testing models: {e}")
        traceback.print_exc()