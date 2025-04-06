"""
Configuration settings for the CPI Analysis & Prediction Dashboard.
Centralizes all configuration parameters for easy maintenance.
"""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = os.path.join(BASE_DIR, "data")

# Data files
INVOICED_JOBS_FILE = os.path.join(DATA_DIR, "invoiced_jobs_this_year_20240912T18_36_36.439126Z.xlsx")
LOST_DEALS_FILE = os.path.join(DATA_DIR, "DealItemReportLOST.xlsx")
ACCOUNT_SEGMENT_FILE = os.path.join(DATA_DIR, "Account+List+with+Segment.csv")

# App configuration
APP_TITLE = "CPI Analysis & Prediction Dashboard"
APP_ICON = "📊"
APP_LAYOUT = "wide"
SIDEBAR_STATE = "expanded"

# Data binning configurations
IR_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
IR_BIN_LABELS = ['0-10', '10-20', '20-30', '30-40', '40-50', 
                '50-60', '60-70', '70-80', '80-90', '90-100']

LOI_BINS = [0, 5, 10, 15, 20, float('inf')]
LOI_BIN_LABELS = ['Very Short (1-5 min)', 'Short (6-10 min)', 
                 'Medium (11-15 min)', 'Long (16-20 min)', 'Very Long (20+ min)']

COMPLETES_BINS = [0, 100, 500, 1000, float('inf')]
COMPLETES_BIN_LABELS = ['Small (1-100)', 'Medium (101-500)', 
                       'Large (501-1000)', 'Very Large (1000+)']

# Visualization settings
# Color-blind friendly palettes
COLORBLIND_PALETTE = {
    'qualitative': ['#3288bd', '#d53e4f', '#66c2a5', '#fee08b', '#e6f598', '#abdda4'],
    'sequential': ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594'],
    'diverging': ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4']
}

# Colors for Won vs Lost (blue-orange contrast for better color-blind accessibility)
WON_COLOR = '#3288bd'  # Blue
LOST_COLOR = '#f58518'  # Orange

# Heatmap color scales (color-blind friendly)
HEATMAP_COLORSCALE_WON = 'Viridis'  # Good color-blind friendly option for sequential data
HEATMAP_COLORSCALE_LOST = 'Plasma'  # Another good color-blind friendly option

# Model configurations
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
HYPERPARAMETER_TUNING = False  # Whether to perform hyperparameter tuning by default

# Cache settings
CACHE_TTL = 3600  # Cache time-to-live in seconds (1 hour)

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# File paths for saving models
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Default model configurations
DEFAULT_MODELS = {
    'Linear Regression': {
        'fit_intercept': True,
        'positive': False
    },
    'Random Forest': {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'random_state': RANDOM_STATE
    },
    'Gradient Boosting': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'random_state': RANDOM_STATE
    }
}

# Feature engineering settings
FEATURE_ENGINEERING_CONFIG = {
    'create_interaction_terms': True,
    'create_log_transforms': True,
    'handle_outliers': True,
    'outlier_threshold': 0.95  # 95th percentile for outlier detection
}

# Dashboard section settings
SHOW_ADVANCED_OPTIONS = False  # Whether to show advanced options by default
DATA_SAMPLE_SIZE = 1000  # Maximum number of rows to display in data tables
