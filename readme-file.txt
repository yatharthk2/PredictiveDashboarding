# CPI Analysis & Prediction Dashboard

This project provides a comprehensive dashboard for analyzing and predicting Cost Per Interview (CPI) pricing for market research projects, with a focus on comparing won and lost bids to optimize pricing strategy.

## Overview

The dashboard consists of three main components:

1. **CPI Analysis Dashboard** - Visualizes the differences between won and lost bids based on key factors (IR, LOI, Sample Size)
2. **CPI Prediction Model** - Uses machine learning to predict optimal CPI based on project parameters
3. **Complete Streamlit Application** - Combines both components into a single integrated dashboard

## Setup Instructions

### Prerequisites

- Python 3.7+ installed
- pip (Python package manager)

### Installation

1. Clone this repository or extract the provided files to your desired location.

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Required Files

Ensure you have the following data files in the same directory as the application:

- `invoiced_jobs_this_year_20240912T18_36_36.439126Z.xlsx` - Contains data on won/invoiced projects
- `DealItemReportLOST.xlsx` - Contains data on lost bids
- `Data Dictionary.xlsx` - Contains information about the data fields (optional, for reference only)
- `AccountListwithSegment.csv` - Contains account segment information (optional, may be used for extended analysis)

## Running the Applications

### Option 1: Complete Dashboard

To run the complete integrated dashboard:

```bash
streamlit run streamlit-app.py
```

This will launch the full application with all components (Overview, CPI Analysis, CPI Prediction, and Insights & Recommendations).

### Option 2: Individual Components

You can also run each component separately:

For the CPI Analysis Dashboard:
```bash
streamlit run cpi-analysis-streamlit.py
```

For the CPI Prediction Model:
```bash
streamlit run cpi-prediction-model.py
```

## Using the Dashboard

The dashboard has four main sections:

1. **Overview** - Provides a high-level summary of key metrics and data distribution
2. **CPI Analysis** - Detailed analysis of CPI by different factors (IR, LOI, Sample Size)
3. **CPI Prediction** - Interactive tool to predict optimal CPI based on input parameters
4. **Insights & Recommendations** - Strategic recommendations based on the analysis

### Filtering Options

- Use the sidebar to toggle filtering of extreme values (>95th percentile)
- Navigate between different sections using the sidebar navigation menu

## Key Features

- **Interactive Visualizations** - All charts are interactive and allow zooming, panning, and hovering for detailed information
- **Machine Learning Models** - Includes multiple prediction models (Linear Regression, Random Forest, Gradient Boosting)
- **Feature Importance Analysis** - Identifies which factors have the most impact on CPI
- **Strategic Recommendations** - Provides actionable insights for pricing strategy

## Customization

You can customize the dashboard by modifying the following aspects:

- **Bin Ranges** - Adjust the IR, LOI, and Sample Size bin ranges in the code to match your business requirements
- **Color Schemes** - Modify the visualization color schemes for better integration with your brand
- **Additional Features** - Extend the analysis by incorporating more features from the provided datasets

## Troubleshooting

If you encounter any issues:

1. Ensure all required data files are in the correct location
2. Verify that column names match those expected in the code (some datasets may have spaces in column names)
3. Check for missing or invalid values in the data
4. Ensure all dependencies are properly installed

## Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)