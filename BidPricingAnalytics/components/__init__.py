"""
Components module for the CPI Analysis & Prediction Dashboard.
This package contains UI components for different dashboard sections.
"""

from .overview import show_overview
from .analysis import show_analysis
from .prediction import show_prediction
from .insights import show_insights

__all__ = [
    'show_overview',
    'show_analysis',
    'show_prediction',
    'show_insights'
]
