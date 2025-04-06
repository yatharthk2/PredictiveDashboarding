"""
CPI Prediction component for the CPI Analysis & Prediction Dashboard.
Provides an interactive tool for predicting optimal CPI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional

# Import ML model utilities
from models.trainer import build_models
from models.predictor import (
    predict_cpi, 
    get_recommendation, 
    get_detailed_pricing_strategy,
    simulate_win_probability
)

# Import visualization utilities
from utils.visualization import (
    create_feature_importance_chart,
    create_prediction_comparison_chart
)

# Import data processing utilities
from utils.data_processor import prepare_model_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def show_prediction(combined_data_engineered: pd.DataFrame, won_data: pd.DataFrame, lost_data: pd.DataFrame) -> None:
    """
    Display the CPI prediction tool for estimating optimal pricing.
    
    Args:
        combined_data_engineered (pd.DataFrame): Engineered DataFrame with features for modeling
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
    """
    try:
        st.title("CPI Prediction Model")
        
        # Introduction
        st.markdown("""
        This tool uses machine learning to predict the optimal CPI (Cost Per Interview) based on:
        - **IR (Incidence Rate)**: The percentage of people who qualify for a survey
        - **LOI (Length of Interview)**: How long the survey takes in minutes
        - **Sample Size**: The number of completed interviews
        
        Enter your project parameters below to get CPI predictions and pricing recommendations.
        """)
        
        # Build models
        with st.spinner("Training prediction models (this may take a moment)..."):
            # Check if we have enough data
            if len(combined_data_engineered) < 10:
                st.warning("Not enough data to build reliable prediction models. Please ensure you have sufficient data.")
                return
            
            # Prepare model data
            X, y = prepare_model_data(combined_data_engineered)
            
            # Check if preparation was successful
            if len(X) == 0 or len(y) == 0:
                st.warning("Failed to prepare model data. Please check your dataset for missing values or data format issues.")
                return
            
            # Build models
            do_tuning = st.sidebar.checkbox("Use advanced model tuning (slower)", value=False)
            trained_models, model_scores, feature_importance = build_models(X, y, do_tuning)
        
        # Toggle for advanced options
        show_advanced = st.sidebar.checkbox("Show advanced model details", value=False)
        
        if show_advanced:
            # Display model metrics in sidebar
            st.sidebar.title("Model Performance")
            for model_name, metrics in model_scores.items():
                st.sidebar.subheader(model_name)
                for metric_name, value in metrics.items():
                    st.sidebar.text(f"{metric_name}: {value:.4f}")
            
            # Show feature importance
            st.header("Feature Importance Analysis")
            
            if len(feature_importance) > 0:
                # Create feature importance chart
                fig = create_feature_importance_chart(feature_importance)
                st.plotly_chart(fig, use_container_width=True)
                
                # Add interpretation
                with st.expander("📊 Interpretation"):
                    st.markdown("""
                    ### Understanding Feature Importance
                    
                    Feature importance shows which factors have the strongest influence on CPI in our model.
                    Longer bars indicate more significant impact on the predicted price.
                    
                    ### Key Insights
                    
                    1. **Primary Drivers**: The top features have the strongest influence on CPI predictions.
                       These should be your primary focus when estimating prices.
                    
                    2. **Relative Importance**: The values represent the relative importance of each feature
                       compared to others. For example, a feature with 0.4 importance has twice the influence
                       of a feature with 0.2 importance.
                    
                    3. **Strategic Focus**: When negotiating or adjusting bids, focus on the top features
                       as they will have the largest impact on competitive pricing.
                    """)
            else:
                st.warning("Feature importance analysis is not available. This may be due to the model type or insufficient data.")
        
        # User input for predictions
        st.header("Predict CPI")
        
        # Create 3 columns for inputs
        col1, col2, col3 = st.columns(3)
        
        # Calculate min, max, and default values from data
        ir_min = max(1, int(combined_data_engineered['IR'].min()))
        ir_max = min(100, int(combined_data_engineered['IR'].max()))
        ir_default = int((ir_min + ir_max) / 2)
        
        loi_min = max(1, int(combined_data_engineered['LOI'].min()))
        loi_max = min(60, int(combined_data_engineered['LOI'].max() * 1.2))  # Add some buffer
        loi_default = int((loi_min + loi_max) / 2)
        
        completes_min = max(10, int(combined_data_engineered['Completes'].min()))
        completes_max = min(2000, int(combined_data_engineered['Completes'].max() * 1.2))  # Add some buffer
        completes_default = int((completes_min + completes_max) / 2)
        
        with col1:
            ir = st.slider(
                "Incidence Rate (%)", 
                min_value=ir_min, 
                max_value=ir_max, 
                value=ir_default,
                help="The percentage of people who qualify for your survey"
            )
        
        with col2:
            loi = st.slider(
                "Length of Interview (min)", 
                min_value=loi_min, 
                max_value=loi_max, 
                value=loi_default,
                help="How long the survey takes to complete in minutes"
            )
        
        with col3:
            completes = st.slider(
                "Sample Size (Completes)", 
                min_value=completes_min, 
                max_value=completes_max, 
                value=completes_default,
                help="The number of completed surveys required"
            )
        
        # Create user input dictionary
        user_input = {
            'IR': ir,
            'LOI': loi,
            'Completes': completes
        }
        
        # Prediction section
        if st.button("Predict CPI", type="primary"):
            with st.spinner("Generating predictions..."):
                # Make predictions
                predictions = predict_cpi(trained_models, user_input, X.columns)
                
                if not predictions:
                    st.error("Failed to generate predictions. Please try different input parameters.")
                    return
                
                # Calculate average prediction
                avg_prediction = sum(predictions.values()) / len(predictions)
                
                # Compare to average CPIs
                won_avg = combined_data_engineered[combined_data_engineered['Type'] == 'Won']['CPI'].mean()
                lost_avg = combined_data_engineered[combined_data_engineered['Type'] == 'Lost']['CPI'].mean()
                
                # Display predictions
                st.subheader("CPI Predictions")
                
                # Create prediction comparison chart
                fig = create_prediction_comparison_chart(predictions, won_avg, lost_avg)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display individual predictions
                cols = st.columns(len(predictions) + 1)
                
                # Display model predictions
                for i, (model_name, pred) in enumerate(predictions.items()):
                    with cols[i]:
                        st.metric(model_name, f"${pred:.2f}")
                
                # Display average prediction
                with cols[-1]:
                    st.metric(
                        "Average Prediction", 
                        f"${avg_prediction:.2f}",
                        delta=f"{((avg_prediction - won_avg) / won_avg * 100):.1f}% vs Won Avg"
                    )
                
                # Display comparison and recommendation
                st.subheader("Pricing Recommendation")
                
                # Create comparison table
                comparison_data = {
                    "Metric": ["Won Bids Average", "Lost Bids Average", "Predicted CPI"],
                    "CPI": [f"${won_avg:.2f}", f"${lost_avg:.2f}", f"${avg_prediction:.2f}"],
                    "Difference vs Won Avg": ["0.0%", f"{((lost_avg - won_avg) / won_avg * 100):.1f}%", f"{((avg_prediction - won_avg) / won_avg * 100):.1f}%"]
                }
                comparison_df = pd.DataFrame(comparison_data)
                
                # Use a different style for the predicted row
                st.dataframe(
                    comparison_df,
                    column_config={
                        "Metric": st.column_config.TextColumn("Metric"),
                        "CPI": st.column_config.TextColumn("CPI"),
                        "Difference vs Won Avg": st.column_config.TextColumn("Difference vs Won Avg")
                    },
                    hide_index=True
                )
                
                # Display recommendation
                recommendation = get_recommendation(avg_prediction, won_avg, lost_avg)
                st.markdown(f"""
                **Recommendation:**
                {recommendation}
                """)
                
                # Win probability simulation
                win_prob = simulate_win_probability(avg_prediction, user_input, won_data, lost_data)
                
                if win_prob:
                    st.metric(
                        "Estimated Win Probability", 
                        f"{win_prob['win_probability']:.1f}%",
                        help="Based on historical win rates at similar price points"
                    )
                
                # Detailed pricing strategy
                st.subheader("Detailed Pricing Strategy")
                strategy = get_detailed_pricing_strategy(avg_prediction, user_input, won_data, lost_data)
                st.markdown(strategy)
                
                # Similar projects analysis
                st.subheader("Similar Projects Analysis")
                
                # Filter for similar projects
                ir_range = 15  # IR range to consider similar
                loi_range = 5   # LOI range to consider similar
                
                similar_won = won_data[
                    (won_data['IR'] >= ir - ir_range) & (won_data['IR'] <= ir + ir_range) &
                    (won_data['LOI'] >= loi - loi_range) & (won_data['LOI'] <= loi + loi_range)
                ]
                
                similar_lost = lost_data[
                    (lost_data['IR'] >= ir - ir_range) & (lost_data['IR'] <= ir + ir_range) &
                    (lost_data['LOI'] >= loi - loi_range) & (lost_data['LOI'] <= loi + loi_range)
                ]
                
                # Create tabs for similar won and lost projects
                sim_tabs = st.tabs(["Similar Won Projects", "Similar Lost Projects"])
                
                with sim_tabs[0]:
                    if len(similar_won) > 0:
                        st.write(f"Found {len(similar_won)} similar won projects with IR from {ir - ir_range} to {ir + ir_range} and LOI from {loi - loi_range} to {loi + loi_range}.")
                        
                        # Show summary stats
                        st.markdown(f"""
                        **Summary Statistics**:
                        - Average CPI: ${similar_won['CPI'].mean():.2f}
                        - Median CPI: ${similar_won['CPI'].median():.2f}
                        - Min CPI: ${similar_won['CPI'].min():.2f}
                        - Max CPI: ${similar_won['CPI'].max():.2f}
                        - Standard Deviation: ${similar_won['CPI'].std():.2f}
                        """)
                        
                        # Show similar projects table with selected columns
                        display_cols = ['CPI', 'IR', 'LOI', 'Completes', 'Revenue']
                        if 'Client' in similar_won.columns:
                            display_cols = ['Client'] + display_cols
                        if 'Date' in similar_won.columns:
                            display_cols.append('Date')
                        
                        st.dataframe(
                            similar_won[display_cols].sort_values('CPI'),
                            column_config={
                                "CPI": st.column_config.NumberColumn("CPI ($)", format="$%.2f"),
                                "IR": st.column_config.NumberColumn("IR (%)", format="%.1f"),
                                "LOI": st.column_config.NumberColumn("LOI (min)", format="%.1f"),
                                "Completes": st.column_config.NumberColumn("Completes", format="%d"),
                                "Revenue": st.column_config.NumberColumn("Revenue ($)", format="$%.2f"),
                            }
                        )
                    else:
                        st.write("No similar won projects found.")
                
                with sim_tabs[1]:
                    if len(similar_lost) > 0:
                        st.write(f"Found {len(similar_lost)} similar lost projects with IR from {ir - ir_range} to {ir + ir_range} and LOI from {loi - loi_range} to {loi + loi_range}.")
                        
                        # Show summary stats
                        st.markdown(f"""
                        **Summary Statistics**:
                        - Average CPI: ${similar_lost['CPI'].mean():.2f}
                        - Median CPI: ${similar_lost['CPI'].median():.2f}
                        - Min CPI: ${similar_lost['CPI'].min():.2f}
                        - Max CPI: ${similar_lost['CPI'].max():.2f}
                        - Standard Deviation: ${similar_lost['CPI'].std():.2f}
                        """)
                        
                        # Show similar projects table with selected columns
                        display_cols = ['CPI', 'IR', 'LOI', 'Completes', 'Revenue']
                        if 'Client' in similar_lost.columns:
                            display_cols = ['Client'] + display_cols
                        if 'Date' in similar_lost.columns:
                            display_cols.append('Date')
                        
                        st.dataframe(
                            similar_lost[display_cols].sort_values('CPI'),
                            column_config={
                                "CPI": st.column_config.NumberColumn("CPI ($)", format="$%.2f"),
                                "IR": st.column_config.NumberColumn("IR (%)", format="%.1f"),
                                "LOI": st.column_config.NumberColumn("LOI (min)", format="%.1f"),
                                "Completes": st.column_config.NumberColumn("Completes", format="%d"),
                                "Revenue": st.column_config.NumberColumn("Revenue ($)", format="$%.2f"),
                            }
                        )
                    else:
                        st.write("No similar lost projects found.")
        
        # Add info section at the bottom
        st.markdown("---")
        st.info("""
        **How to use this tool**: 
        
        1. Adjust the sliders to set your project parameters (IR, LOI, Sample Size)
        2. Click "Predict CPI" to generate predictions and recommendations
        3. Review the predicted CPI values from different models
        4. Use the detailed pricing strategy to guide your bid decision
        5. Explore similar projects for additional context
        """)
    
    except Exception as e:
        logger.error(f"Error in prediction component: {e}", exc_info=True)
        st.error(f"An error occurred while rendering the prediction component: {str(e)}")
