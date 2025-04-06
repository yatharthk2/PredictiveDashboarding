"""
Utilities module for the CPI Analysis & Prediction Dashboard.
This package contains utility functions for data processing and visualization.
"""

from .data_loader import load_data
from .data_processor import (
    apply_all_bins, 
    create_ir_bins, 
    create_loi_bins, 
    create_completes_bins,
    engineer_features,
    prepare_model_data,
    get_data_summary
)
from .visualization import (
    create_type_distribution_chart,
    create_cpi_distribution_boxplot,
    create_cpi_histogram_comparison,
    create_cpi_vs_ir_scatter,
    create_bar_chart_by_bin,
    create_heatmap,
    create_feature_importance_chart,
    create_prediction_comparison_chart,
    create_cpi_efficiency_chart
)

__all__ = [
    'load_data',
    'apply_all_bins',
    'create_ir_bins',
    'create_loi_bins',
    'create_completes_bins',
    'engineer_features',
    'prepare_model_data',
    'get_data_summary',
    'create_type_distribution_chart',
    'create_cpi_distribution_boxplot',
    'create_cpi_histogram_comparison',
    'create_cpi_vs_ir_scatter',
    'create_bar_chart_by_bin',
    'create_heatmap',
    'create_feature_importance_chart',
    'create_prediction_comparison_chart',
    'create_cpi_efficiency_chart'
]
