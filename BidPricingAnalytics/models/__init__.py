"""
Models module for the CPI Analysis & Prediction Dashboard.
This package contains machine learning model functionality.
"""

from .trainer import (
    build_models,
    build_models_default,
    build_models_with_tuning,
    save_models,
    save_model_pipeline,
    load_models,
    cross_validate_models,
    evaluate_model_assumptions
)
from .predictor import (
    predict_cpi,
    get_recommendation,
    get_detailed_pricing_strategy,
    simulate_win_probability,
    get_prediction_metrics
)

__all__ = [
    'build_models',
    'build_models_default',
    'build_models_with_tuning',
    'save_models',
    'save_model_pipeline',
    'load_models',
    'cross_validate_models',
    'evaluate_model_assumptions',
    'predict_cpi',
    'get_recommendation',
    'get_detailed_pricing_strategy',
    'simulate_win_probability',
    'get_prediction_metrics'
]