"""
Models module for the CPI Analysis & Prediction Dashboard.
This package contains machine learning model functionality.
"""

from .trainer import (
    build_models,
    save_models,
    load_models,
    cross_validate_models
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
    'save_models',
    'load_models',
    'cross_validate_models',
    'predict_cpi',
    'get_recommendation',
    'get_detailed_pricing_strategy',
    'simulate_win_probability',
    'get_prediction_metrics'
]
