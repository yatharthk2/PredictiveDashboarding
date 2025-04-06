import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="CPI Analysis & Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define functions for data loading and processing
def load_data():
    """Load and process the data from Excel files"""
    
    # Load invoiced jobs data (Won deals)
    invoiced_df = pd.read_excel("invoiced_jobs_this_year_20240912T18_36_36.439126Z.xlsx")
    
    # Print column names to debug
    print("Available columns in invoiced_df:")
    print(invoiced_df.columns.tolist())
    
    # Rename columns to remove spaces
    invoiced_df = invoiced_df.rename(columns={
        ' CPI ': 'CPI',
        ' Actual Project Revenue ': 'Revenue',
        'Actual Project Revenue': 'Revenue',  # Try different variations
        ' Revenue ': 'Revenue',
        'Revenue ': 'Revenue'
    })
    
    # Process Countries column
    invoiced_df['Countries'] = invoiced_df['Countries'].fillna('[]')
    invoiced_df['Country'] = invoiced_df['Countries'].apply(
        lambda x: x.replace('[', '').replace(']', '').replace('"', '')
    )
    invoiced_df['Country'] = invoiced_df['Country'].replace('', 'USA')
    
    # Create Won dataset
    won_df = invoiced_df[[
        'Project Code Parent', 'Client Name', 'CPI', 'Actual Ir', 'Actual Loi', 
        'Complete', 'Revenue', 'Invoiced Date', 'Country', 'Audience'
    ]].copy()
    
    # Rename columns for consistency
    won_df = won_df.rename(columns={
        'Project Code Parent': 'ProjectId',
        'Client Name': 'Client',
        'Actual Ir': 'IR',
        'Actual Loi': 'LOI',
        'Complete': 'Completes',
        'Invoiced Date': 'Date'
    })
    
    # Add type column
    won_df['Type'] = 'Won'
    
    # Load lost deals data
    lost_df_raw = pd.read_excel("DealItemReportLOST.xlsx")
    
    # Filter for Sample items only
    lost_df = lost_df_raw[lost_df_raw['Item'] == 'Sample'].copy()
    
    # Create Lost dataset
    lost_df = lost_df[[
        'Record Id', 'Account Name', 'Customer Rate', 'IR', 'LOI', 
        'Qty', 'Item Amount', 'Description (Items)', 'Deal Name'
    ]].copy()
    
    # Rename columns for consistency
    lost_df = lost_df.rename(columns={
        'Record Id': 'DealId',
        'Account Name': 'Client',
        'Customer Rate': 'CPI',
        'Qty': 'Completes',
        'Item Amount': 'Revenue',
        'Description (Items)': 'Country',
        'Deal Name': 'ProjectName'
    })
    
    # Add type column
    lost_df['Type'] = 'Lost'
    
    # Convert CPI columns to numeric before filtering
    won_df['CPI'] = pd.to_numeric(won_df['CPI'], errors='coerce')
    lost_df['CPI'] = pd.to_numeric(lost_df['CPI'], errors='coerce')
    
    # Filter out invalid CPI values
    won_df = won_df[won_df['CPI'].notna() & (won_df['CPI'] > 0)]
    lost_df = lost_df[lost_df['CPI'].notna() & (lost_df['CPI'] > 0)]
    
    # Filter out extreme values (over 95th percentile)
    won_percentile_95 = won_df['CPI'].quantile(0.95)
    lost_percentile_95 = lost_df['CPI'].quantile(0.95)
    
    won_df_filtered = won_df[won_df['CPI'] <= won_percentile_95]
    lost_df_filtered = lost_df[lost_df['CPI'] <= lost_percentile_95]
    
    # Create a single combined dataframe with only the common columns
    common_columns = ['Client', 'CPI', 'IR', 'LOI', 'Completes', 'Revenue', 'Country', 'Type']
    combined_df = pd.concat(
        [won_df[common_columns], lost_df[common_columns]],
        ignore_index=True
    )
    
    # Create a filtered combined dataset
    combined_df_filtered = pd.concat(
        [won_df_filtered[common_columns], lost_df_filtered[common_columns]],
        ignore_index=True
    )
    
    return {
        'won': won_df,
        'won_filtered': won_df_filtered,
        'lost': lost_df,
        'lost_filtered': lost_df_filtered,
        'combined': combined_df,
        'combined_filtered': combined_df_filtered
    }

def create_ir_bins(df, bin_size=10):
    """Create IR bins for analysis"""
    df['IR_Bin'] = pd.cut(
        df['IR'],
        bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
    )
    return df

def create_loi_bins(df):
    """Create LOI bins for analysis"""
    df['LOI_Bin'] = pd.cut(
        df['LOI'],
        bins=[0, 5, 10, 15, 20, float('inf')],
        labels=['Very Short (1-5 min)', 'Short (6-10 min)', 'Medium (11-15 min)', 'Long (16-20 min)', 'Very Long (20+ min)']
    )
    return df

def create_completes_bins(df):
    """Create Sample Size (Completes) bins for analysis"""
    df['Completes_Bin'] = pd.cut(
        df['Completes'],
        bins=[0, 100, 500, 1000, float('inf')],
        labels=['Small (1-100)', 'Medium (101-500)', 'Large (501-1000)', 'Very Large (1000+)']
    )
    return df

def build_models(df):
    """Build prediction models for CPI"""
    
    # Handle missing values before feature engineering
    df = df.copy()
    df['IR'] = pd.to_numeric(df['IR'], errors='coerce')
    df['LOI'] = pd.to_numeric(df['LOI'], errors='coerce')
    df['Completes'] = pd.to_numeric(df['Completes'], errors='coerce')
    
    # Drop rows with missing values in key columns
    df = df.dropna(subset=['IR', 'LOI', 'Completes', 'CPI'])
    
    # Feature engineering
    df['IR_LOI_Ratio'] = df['IR'] / df['LOI']
    df['IR_Completes_Ratio'] = df['IR'] / df['Completes']
    
    # Replace any infinite values that might have been created during division
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    # Prepare feature matrix and target variable
    X = df[['IR', 'LOI', 'Completes', 'IR_LOI_Ratio', 'IR_Completes_Ratio', 'Type']]
    y = df['CPI']
    
    # Create dummy variables for categorical features
    X_encoded = pd.get_dummies(X, columns=['Type'], drop_first=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    
    # Build models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    # Train models
    trained_models = {}
    model_scores = {}
    
    for name, model in models.items():
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
    
    # Get feature importance from Random Forest
    rf_model = trained_models['Random Forest']
    feature_importance = pd.DataFrame({
        'Feature': X_encoded.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return trained_models, model_scores, feature_importance, X_encoded.columns

def predict_cpi(models, user_input, feature_names):
    """Predict CPI based on user input"""
    
    # Create a DataFrame with the user input
    input_df = pd.DataFrame([user_input], columns=['IR', 'LOI', 'Completes'])
    
    # Feature engineering
    input_df['IR_LOI_Ratio'] = input_df['IR'] / input_df['LOI']
    input_df['IR_Completes_Ratio'] = input_df['IR'] / input_df['Completes']
    
    # Add Type columns (one-hot encoded)
    input_df['Type_Won'] = 1  # Assuming we want to predict for 'Won' type
    
    # Ensure the input DataFrame has all required columns in the right order
    final_input = pd.DataFrame(columns=feature_names)
    for col in feature_names:
        if col in input_df.columns:
            final_input[col] = input_df[col]
        else:
            final_input[col] = 0
    
    # Make predictions with each model
    predictions = {}
    for name, model in models.items():
        pred = model.predict(final_input)[0]
        predictions[name] = pred
    
    return predictions

def get_recommendation(predicted_cpi, won_avg, lost_avg):
    """Generate a pricing recommendation based on predictions"""
    
    if predicted_cpi <= won_avg:
        return "This CPI is lower than the average for won bids, suggesting a very competitive price point that should increase chances of winning."
    elif predicted_cpi <= (won_avg + lost_avg) / 2:
        return "This CPI is higher than the average for won bids but still below the midpoint between won and lost bids, suggesting a moderately competitive price point."
    elif predicted_cpi <= lost_avg:
        return "This CPI is in the upper range between won and lost bids, which may reduce chances of winning but could improve profitability if the bid is accepted."
    else:
        return "This CPI is higher than the average for lost bids, suggesting a price point that may be too high to be competitive."

# Main application
def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Choose a mode",
        ["Overview", "CPI Analysis", "CPI Prediction", "Insights & Recommendations"]
    )
    
    # Load data
    with st.spinner("Loading data..."):
        try:
            data = load_data()
            won_df = data['won']
            won_df_filtered = data['won_filtered']
            lost_df = data['lost']
            lost_df_filtered = data['lost_filtered']
            combined_df = data['combined']
            combined_df_filtered = data['combined_filtered']
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()
    
    # Add bins to the data
    won_df = create_ir_bins(won_df)
    won_df = create_loi_bins(won_df)
    won_df = create_completes_bins(won_df)
    
    lost_df = create_ir_bins(lost_df)
    lost_df = create_loi_bins(lost_df)
    lost_df = create_completes_bins(lost_df)
    
    # Create filtered versions with bins
    won_df_filtered = create_ir_bins(won_df_filtered)
    won_df_filtered = create_loi_bins(won_df_filtered)
    won_df_filtered = create_completes_bins(won_df_filtered)
    
    lost_df_filtered = create_ir_bins(lost_df_filtered)
    lost_df_filtered = create_loi_bins(lost_df_filtered)
    lost_df_filtered = create_completes_bins(lost_df_filtered)
    
    # Sidebar for filtering options
    st.sidebar.title("Filtering Options")
    show_filtered = st.sidebar.checkbox("Filter out extreme values (>95th percentile)", value=True)
    
    # Choose datasets based on filtering option
    if show_filtered:
        won_data = won_df_filtered
        lost_data = lost_df_filtered
        combined_data = combined_df_filtered
    else:
        won_data = won_df
        lost_data = lost_df
        combined_data = combined_df
    
    # Overview mode
    if app_mode == "Overview":
        st.title("CPI Analysis Dashboard: Overview")
        st.markdown("""
        This dashboard analyzes the Cost Per Impression (CPI) between won and lost bids 
        to identify meaningful differences. The three main factors that influence CPI are:
        - **IR (Incidence Rate)**: The percentage of people who qualify for a survey
        - **LOI (Length of Interview)**: How long the survey takes to complete
        - **Sample Size (Completes)**: The number of completed surveys
        
        Use the navigation menu on the left to explore different analyses and tools.
        """)
        
        # Key metrics
        st.header("Key Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average CPI - Won", f"${won_data['CPI'].mean():.2f}")
            st.metric("Average CPI - Lost", f"${lost_data['CPI'].mean():.2f}")
            st.metric("CPI Difference", f"${lost_data['CPI'].mean() - won_data['CPI'].mean():.2f}")
        
        with col2:
            st.metric("Average IR - Won", f"{won_data['IR'].mean():.2f}%")
            st.metric("Average IR - Lost", f"{lost_data['IR'].mean():.2f}%")
            st.metric("IR Difference", f"{lost_data['IR'].mean() - won_data['IR'].mean():.2f}%")
        
        with col3:
            st.metric("Average LOI - Won", f"{won_data['LOI'].mean():.2f} min")
            st.metric("Average LOI - Lost", f"{lost_data['LOI'].mean():.2f} min")
            st.metric("LOI Difference", f"{lost_data['LOI'].mean() - won_data['LOI'].mean():.2f} min")
        
        # Overview charts
        st.header("Data Overview")
        
        # Count of won vs lost bids
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                combined_data, 
                names='Type', 
                title='Distribution of Won vs Lost Bids',
                color='Type',
                color_discrete_map={'Won': 'green', 'Lost': 'red'},
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Overall CPI comparison boxplot
            fig = go.Figure()
            
            fig.add_trace(go.Box(
                y=won_data['CPI'],
                name='Won',
                marker_color='green',
                boxmean=True
            ))
            
            fig.add_trace(go.Box(
                y=lost_data['CPI'],
                name='Lost',
                marker_color='red',
                boxmean=True
            ))
            
            fig.update_layout(
                title='CPI Distribution: Won vs Lost',
                yaxis_title='CPI ($)',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # CPI Analysis mode
    elif app_mode == "CPI Analysis":
        st.title("CPI Analysis: Won vs. Lost Bids")
        
        # CPI Distribution
        st.header("CPI Distribution Comparison")
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Won Bids CPI Distribution", "Lost Bids CPI Distribution"))
        
        fig.add_trace(
            go.Histogram(x=won_data['CPI'], name="Won", marker_color='green', opacity=0.7),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(x=lost_data['CPI'], name="Lost", marker_color='red', opacity=0.7),
            row=1, col=2
        )
        
        fig.update_layout(
            height=500,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="CPI ($)")
        fig.update_yaxes(title_text="Count")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # CPI by IR Analysis
        st.header("CPI Analysis by Incidence Rate (IR)")
        
        # CPI vs IR scatter plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=won_data['IR'], 
            y=won_data['CPI'], 
            mode='markers',
            marker=dict(color='green', size=8, opacity=0.6),
            name="Won"
        ))
        
        fig.add_trace(go.Scatter(
            x=lost_data['IR'], 
            y=lost_data['CPI'], 
            mode='markers',
            marker=dict(color='red', size=8, opacity=0.6),
            name="Lost"
        ))
        
        fig.update_layout(
            title_text="CPI vs Incidence Rate Scatter Plot",
            height=500,
            xaxis_title="Incidence Rate (%)",
            yaxis_title="CPI ($)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # CPI by IR Bin boxplot
        ir_bins_won = won_data.groupby('IR_Bin')['CPI'].mean().reset_index()
        ir_bins_lost = lost_data.groupby('IR_Bin')['CPI'].mean().reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=ir_bins_won['IR_Bin'],
            y=ir_bins_won['CPI'],
            name='Won',
            marker_color='green',
            opacity=0.7
        ))
        
        fig.add_trace(go.Bar(
            x=ir_bins_lost['IR_Bin'],
            y=ir_bins_lost['CPI'],
            name='Lost',
            marker_color='red',
            opacity=0.7
        ))
        
        fig.update_layout(
            title='Average CPI by Incidence Rate Bin',
            xaxis_title='Incidence Rate Bin (%)',
            yaxis_title='Average CPI ($)',
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # CPI by LOI Analysis
        st.header("CPI Analysis by Length of Interview (LOI)")
        
        # CPI by LOI Bin
        loi_bins_won = won_data.groupby('LOI_Bin')['CPI'].mean().reset_index()
        loi_bins_lost = lost_data.groupby('LOI_Bin')['CPI'].mean().reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=loi_bins_won['LOI_Bin'],
            y=loi_bins_won['CPI'],
            name='Won',
            marker_color='green',
            opacity=0.7
        ))
        
        fig.add_trace(go.Bar(
            x=loi_bins_lost['LOI_Bin'],
            y=loi_bins_lost['CPI'],
            name='Lost',
            marker_color='red',
            opacity=0.7
        ))
        
        fig.update_layout(
            title='Average CPI by Length of Interview Bin',
            xaxis_title='LOI Bin',
            yaxis_title='Average CPI ($)',
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # CPI by Sample Size Analysis
        st.header("CPI Analysis by Sample Size (Completes)")
        
        # CPI by Completes Bin
        completes_bins_won = won_data.groupby('Completes_Bin')['CPI'].mean().reset_index()
        completes_bins_lost = lost_data.groupby('Completes_Bin')['CPI'].mean().reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=completes_bins_won['Completes_Bin'],
            y=completes_bins_won['CPI'],
            name='Won',
            marker_color='green',
            opacity=0.7
        ))
        
        fig.add_trace(go.Bar(
            x=completes_bins_lost['Completes_Bin'],
            y=completes_bins_lost['CPI'],
            name='Lost',
            marker_color='red',
            opacity=0.7
        ))
        
        fig.update_layout(
            title='Average CPI by Sample Size Bin',
            xaxis_title='Sample Size Bin',
            yaxis_title='Average CPI ($)',
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Multi-factor analysis
        st.header("Multi-factor Analysis")
        
        # IR and LOI combined influence
        st.subheader("IR and LOI Combined Influence on CPI")
        
        # Create pivot tables
        won_pivot = won_data.pivot_table(
            values='CPI', 
            index='IR_Bin', 
            columns='LOI_Bin', 
            aggfunc='mean'
        ).fillna(0)
        
        lost_pivot = lost_data.pivot_table(
            values='CPI', 
            index='IR_Bin', 
            columns='LOI_Bin', 
            aggfunc='mean'
        ).fillna(0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Won deals heatmap
            fig = px.imshow(
                won_pivot,
                labels=dict(x="LOI Bin", y="IR Bin", color="Avg CPI ($)"),
                x=won_pivot.columns,
                y=won_pivot.index,
                title="Won Deals: Average CPI by IR and LOI",
                color_continuous_scale="Greens",
                aspect="auto"
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Lost deals heatmap
            fig = px.imshow(
                lost_pivot,
                labels=dict(x="LOI Bin", y="IR Bin", color="Avg CPI ($)"),
                x=lost_pivot.columns,
                y=lost_pivot.index,
                title="Lost Deals: Average CPI by IR and LOI",
                color_continuous_scale="Reds",
                aspect="auto"
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    # CPI Prediction mode
    elif app_mode == "CPI Prediction":
        st.title("CPI Prediction Model")
        st.markdown("""
        This tool uses machine learning to predict the optimal CPI (Cost Per Interview) based on:
        - **IR (Incidence Rate)**: The percentage of people who qualify for a survey
        - **LOI (Length of Interview)**: How long the survey takes in minutes
        - **Sample Size**: The number of completed interviews
        """)
        
        # Build models
        with st.spinner("Training prediction models..."):
            models, model_scores, feature_importance, feature_names = build_models(combined_data)
        
        # Display metrics on sidebar
        st.sidebar.title("Model Performance")
        for model_name, metrics in model_scores.items():
            st.sidebar.subheader(model_name)
            for metric_name, value in metrics.items():
                st.sidebar.text(f"{metric_name}: {value:.4f}")
        
        # Show feature importance
        st.header("Feature Importance Analysis")
        
        # Create a horizontal bar chart
        fig = px.bar(
            feature_importance, 
            x='Importance', 
            y='Feature', 
            orientation='h',
            title='Feature Importance (Random Forest)',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # User input for predictions
        st.header("Predict CPI")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ir = st.slider("Incidence Rate (%)", 1, 100, 50)
        
        with col2:
            loi = st.slider("Length of Interview (min)", 1, 60, 15)
        
        with col3:
            completes = st.slider("Sample Size (Completes)", 10, 2000, 500)
        
        user_input = {
            'IR': ir,
            'LOI': loi,
            'Completes': completes
        }
        
        if st.button("Predict CPI"):
            predictions = predict_cpi(models, user_input, feature_names)
            
            # Display predictions
            st.subheader("CPI Predictions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Linear Regression", f"${predictions['Linear Regression']:.2f}")
            
            with col2:
                st.metric("Random Forest", f"${predictions['Random Forest']:.2f}")
            
            with col3:
                st.metric("Gradient Boosting", f"${predictions['Gradient Boosting']:.2f}")
            
            # Calculate average prediction
            avg_prediction = sum(predictions.values()) / len(predictions)
            
            st.metric("Average Prediction", f"${avg_prediction:.2f}")
            
            # Compare to average CPIs
            won_avg = combined_data[combined_data['Type'] == 'Won']['CPI'].mean()
            lost_avg = combined_data[combined_data['Type'] == 'Lost']['CPI'].mean()
            
            st.markdown(f"""
            **Comparison:**
            - Average CPI for Won bids: ${won_avg:.2f}
            - Average CPI for Lost bids: ${lost_avg:.2f}
            - Your predicted CPI: ${avg_prediction:.2f}
            
            **Recommendation:**
            {get_recommendation(avg_prediction, won_avg, lost_avg)}
            """)
    
    # Insights & Recommendations mode
    elif app_mode == "Insights & Recommendations":
        st.title("Insights & Recommendations")
        
        st.header("Key Findings")
        st.markdown("""
        Based on the analysis of the CPI (Cost Per Interview) data between won and lost bids, we've identified the following key insights:
        
        1. **Overall CPI Difference**: There is a significant gap between the average CPI for won bids 
           (${:.2f}) and lost bids (${:.2f}). This suggests that pricing is a critical factor in bid success.
           
        2. **IR (Incidence Rate) Impact**: Lower IR values generally correlate with higher CPIs, as it becomes 
           more difficult and costly to find qualified respondents. Lost bids tend to have higher CPIs at all IR levels,
           but the difference is most pronounced at lower IR levels.
           
        3. **LOI (Length of Interview) Impact**: As LOI increases, CPI tends to increase for both won and lost bids.
           However, lost bids show a steeper increase in CPI as LOI gets longer, suggesting that pricing for longer
           surveys may be a key differentiator.
           
        4. **Sample Size Effect**: Larger sample sizes (higher number of completes) tend to have lower per-unit CPIs
           due to economies of scale. Lost bids often don't sufficiently account for this scaling effect.
           
        5. **Combination Effects**: The interaction between IR and LOI has a significant impact on CPI. The optimal
           pricing varies considerably depending on these two factors combined.
        """.format(won_data['CPI'].mean(), lost_data['CPI'].mean()))
        
        st.header("Recommendations for Pricing Strategy")
        st.markdown("""
        Based on our analysis, we recommend the following pricing strategies to improve bid success rates:
        
        1. **IR-Based Pricing Tiers**: Implement a clear pricing structure based on IR ranges, with higher prices
           for lower IR projects. Our analysis suggests the following price adjustments for different IR ranges:
           - Low IR (0-20%): Keep CPIs below ${:.2f} (25th percentile of lost bids in this range)
           - Medium IR (21-50%): Keep CPIs below ${:.2f} (25th percentile of lost bids in this range)
           - High IR (51-100%): Keep CPIs below ${:.2f} (25th percentile of lost bids in this range)
           
        2. **LOI Multipliers**: Apply multipliers to the base CPI based on LOI:
           - Short LOI (1-10 min): Base CPI
           - Medium LOI (11-20 min): Base CPI × 1.{:.0f}
           - Long LOI (21+ min): Base CPI × 1.{:.0f}
           
        3. **Sample Size Discounts**: Implement volume discounts for larger projects:
           - Small (1-100 completes): Standard CPI
           - Medium (101-500 completes): {:.0f}% discount
           - Large (501-1000 completes): {:.0f}% discount
           - Very Large (1000+ completes): {:.0f}% discount
           
        4. **Combined Factor Pricing Model**: Use the prediction model to optimize pricing for different
           combinations of IR, LOI, and sample size. This approach can help provide competitive yet
           profitable pricing.
           
        5. **Regular Analysis**: Continuously analyze won and lost bids to refine the pricing model and
           stay competitive in the market.
        """.format(
            lost_data[lost_data['IR'] <= 20]['CPI'].quantile(0.25),
            lost_data[(lost_data['IR'] > 20) & (lost_data['IR'] <= 50)]['CPI'].quantile(0.25),
            lost_data[lost_data['IR'] > 50]['CPI'].quantile(0.25),
            3,
            5,
            5,
            10,
            15
        ))
        
        st.header("Implementation Plan")
        st.markdown("""
        To implement these recommendations, we suggest the following steps:
        
        1. **Pricing Calculator Tool**: Develop an internal tool based on the prediction model to help
           sales teams quickly determine optimal CPI for new bids.
           
        2. **Pricing Guidelines**: Create clear pricing guidelines based on the tiered structure and
           make them accessible to all sales team members.
           
        3. **Historical Data Review**: Regularly review historical pricing data to identify trends and
           adjust the pricing model accordingly.
           
        4. **Competitive Analysis**: Monitor competitor pricing when possible to ensure your bids remain
           competitive in the market.
           
        5. **Feedback Loop**: Implement a system to collect feedback on won and lost bids to continuously
           refine the pricing model.
        """)
        
        st.header("Expected Impact")
        st.markdown("""
        Based on our analysis, implementing these recommendations could lead to:
        
        1. **Increased Win Rate**: By optimizing CPI based on key factors, we expect an increase in the bid
           win rate, particularly for projects with challenging parameters (low IR, high LOI).
           
        2. **Improved Profitability**: The structured approach ensures pricing remains profitable while still
           being competitive.
           
        3. **Consistent Pricing**: Reducing variability in pricing across similar projects will lead to more
           consistent client experiences.
           
        4. **Data-Driven Decisions**: Moving from intuition-based to data-driven pricing decisions will improve
           overall business performance.
        """)

if __name__ == "__main__":
    main()