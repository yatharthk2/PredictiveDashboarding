        
        # Final pricing recommendation based on all factors
        total_adjustment = 0
        
        # IR adjustment
        if ir < 20:
            total_adjustment += 17.5  # Midpoint of 15-20%
        elif ir < 50:
            total_adjustment += 7.5   # Midpoint of 5-10%
        
        # LOI adjustment
        if loi > 20:
            total_adjustment += 17.5  # Midpoint of 15-20%
        elif loi > 10:
            total_adjustment += 7.5   # Midpoint of 5-10%
        
        # Sample size adjustment
        if completes > 1000:
            total_adjustment -= 17.5  # Midpoint of 15-20%
        elif completes > 500:
            total_adjustment -= 12.5  # Midpoint of 10-15%
        elif completes > 100:
            total_adjustment -= 7.5   # Midpoint of 5-10%
        
        # Calculate adjusted CPI
        base_cpi = predicted_cpi
        adjusted_cpi = base_cpi * (1 + total_adjustment / 100)
        
        strategy.append("")
        strategy.append(f"**Net Adjustment:** {total_adjustment:+.1f}%")
        strategy.append(f"**Base CPI:** ${base_cpi:.2f}")
        strategy.append(f"**Recommended CPI:** ${adjusted_cpi:.2f}")
        
        # Add competitive analysis
        strategy.append("")
        strategy.append("**Competitive Positioning:**")
        
        if len(similar_won) > 0 and len(similar_lost) > 0:
            won_avg = similar_won['CPI'].mean()
            lost_avg = similar_lost['CPI'].mean()
            
            if adjusted_cpi <= won_avg:
                strategy.append(f"- This recommended CPI (${adjusted_cpi:.2f}) is below the average of similar won bids (${won_avg:.2f}), suggesting a highly competitive position.")
            elif adjusted_cpi <= (won_avg + lost_avg) / 2:
                strategy.append(f"- This recommended CPI (${adjusted_cpi:.2f}) is above the average of similar won bids (${won_avg:.2f}) but below the midpoint, suggesting a moderately competitive position.")
            elif adjusted_cpi <= lost_avg:
                strategy.append(f"- This recommended CPI (${adjusted_cpi:.2f}) is closer to the average of similar lost bids (${lost_avg:.2f}), which may reduce win probability. Consider additional value-add services to justify this premium.")
            else:
                strategy.append(f"- This recommended CPI (${adjusted_cpi:.2f}) is above the average of similar lost bids (${lost_avg:.2f}), suggesting it may be too high to be competitive unless there are unique selling points.")
        
        # Join the strategy text with line breaks
        return "\n".join(strategy)
    
    except Exception as e:
        logger.error(f"Error in get_detailed_pricing_strategy: {e}", exc_info=True)
        return "Unable to generate detailed pricing strategy due to an error."

def simulate_win_probability(predicted_cpi: float, user_input: Dict[str, float],
                          won_data: pd.DataFrame, lost_data: pd.DataFrame) -> Dict[str, float]:
    """
    Simulate the probability of winning a bid based on the predicted CPI.
    
    Args:
        predicted_cpi (float): Predicted CPI value
        user_input (Dict[str, float]): Dictionary of user input values
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
    
    Returns:
        Dict[str, float]: Dictionary of win probability metrics
    """
    try:
        # Combine won and lost data
        combined_data = pd.concat([
            won_data.assign(Won=1),
            lost_data.assign(Won=0)
        ], ignore_index=True)
        
        # Calculate CPI percentiles
        percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        cpi_percentiles = {p: combined_data['CPI'].quantile(p/100) for p in percentiles}
        
        # Find where the predicted CPI falls in the distribution
        cpi_percentile = 0
        for p in sorted(percentiles):
            if predicted_cpi <= cpi_percentiles[p]:
                cpi_percentile = p
                break
        
        if cpi_percentile == 0:
            cpi_percentile = 100  # Above all percentiles
        
        # Calculate win probability based on percentile
        # Lower percentile (lower price) = higher win probability
        wins_by_percentile = {}
        for p in percentiles:
            below_p = combined_data[combined_data['CPI'] <= combined_data['CPI'].quantile(p/100)]
            if len(below_p) > 0:
                wins_by_percentile[p] = below_p['Won'].mean() * 100
            else:
                wins_by_percentile[p] = 0
        
        # Get win probability for the predicted CPI
        if cpi_percentile in percentiles:
            win_probability = wins_by_percentile[cpi_percentile]
        else:
            # If predicted CPI is above all percentiles, use lowest win rate
            win_probability = min(wins_by_percentile.values())
        
        # Return results
        return {
            'cpi_percentile': cpi_percentile,
            'win_probability': win_probability,
            'percentile_data': {
                'percentiles': percentiles,
                'cpi_values': list(cpi_percentiles.values()),
                'win_rates': list(wins_by_percentile.values())
            }
        }
    
    except Exception as e:
        logger.error(f"Error in simulate_win_probability: {e}", exc_info=True)
        return {}

if __name__ == "__main__":
    # Test prediction functions with sample data
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample models
        class DummyModel:
            def predict(self, X):
                # Simple prediction based on IR and LOI
                return np.array([X['IR'].values[0] * 0.1 + X['LOI'].values[0] * 0.3])
        
        models = {
            'Linear Regression': DummyModel(),
            'Random Forest': DummyModel(),
            'Gradient Boosting': DummyModel()
        }
        
        # Create sample user input
        user_input = {
            'IR': 30,
            'LOI': 15,
            'Completes': 500
        }
        
        # Define feature names
        feature_names = ['IR', 'LOI', 'Completes', 'IR_LOI_Ratio', 
                         'IR_Completes_Ratio', 'Type_Won']
        
        # Test predict_cpi
        print("Testing predict_cpi...")
        predictions = predict_cpi(models, user_input, feature_names)
        print(f"Predictions: {predictions}")
        
        # Test get_prediction_metrics
        print("\nTesting get_prediction_metrics...")
        metrics = get_prediction_metrics(predictions)
        print(f"Metrics: {metrics}")
        
        # Test get_recommendation
        print("\nTesting get_recommendation...")
        recommendation = get_recommendation(7.5, 6.0, 9.0)
        print(f"Recommendation: {recommendation}")
        
        # Create sample won and lost data
        np.random.seed(42)
        n_samples = 50
        
        won_data = pd.DataFrame({
            'CPI': np.random.uniform(5, 8, n_samples),
            'IR': np.random.uniform(20, 80, n_samples),
            'LOI': np.random.uniform(5, 25, n_samples),
            'Completes': np.random.uniform(100, 1000, n_samples)
        })
        
        lost_data = pd.DataFrame({
            'CPI': np.random.uniform(7, 10, n_samples),
            'IR': np.random.uniform(10, 70, n_samples),
            'LOI': np.random.uniform(10, 30, n_samples),
            'Completes': np.random.uniform(50, 800, n_samples)
        })
        
        # Test get_detailed_pricing_strategy
        print("\nTesting get_detailed_pricing_strategy...")
        strategy = get_detailed_pricing_strategy(7.5, user_input, won_data, lost_data)
        print(strategy)
        
        # Test simulate_win_probability
        print("\nTesting simulate_win_probability...")
        win_prob = simulate_win_probability(7.5, user_input, won_data, lost_data)
        print(f"Win Probability: {win_prob['win_probability']:.1f}%")
        print(f"CPI Percentile: {win_prob['cpi_percentile']}")
        
        print("\nAll tests completed successfully")
        
    except Exception as e:
        print(f"Error testing predictor: {e}")
 duration.")
        elif loi > 10:
            strategy.append("- **Medium LOI Adjustment:** Add a premium of 5-10% to the base CPI.")
        else:
            strategy.append("- **Short LOI Adjustment:** No LOI premium needed as the survey is short.")
        
        # Sample size adjustment
        if completes > 1000:
            strategy.append("- **Large Sample Discount:** Offer a 15-20% volume discount due to the very large sample size.")
        elif completes > 500:
            strategy.append("- **Medium Sample Discount:** Offer a 10-15% volume discount due to the large sample size.")
        elif completes > 100:
            strategy.append("- **Small Sample Discount:** Offer a 5-10% volume discount.")
        else:
            strategy.append("- **No Sample Discount:** The sample size is too small to qualify for a volume discount.")
        """
ML model prediction functionality for the CPI Analysis & Prediction Dashboard.
Includes functions for making predictions and generating recommendations.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def predict_cpi(models: Dict[str, Any], user_input: Dict[str, float], feature_names: List[str]) -> Dict[str, float]:
    """
    Predict CPI based on user input.
    
    Args:
        models (Dict[str, Any]): Dictionary of trained models
        user_input (Dict[str, float]): Dictionary of user input values
        feature_names (List[str]): List of feature names expected by the models
    
    Returns:
        Dict[str, float]: Dictionary of model predictions
    """
    try:
        logger.info(f"Making predictions with input: {user_input}")
        
        # Create a DataFrame with the user input
        input_df = pd.DataFrame([user_input], columns=['IR', 'LOI', 'Completes'])
        
        # Feature engineering
        input_df['IR_LOI_Ratio'] = input_df['IR'] / input_df['LOI']
        input_df['IR_Completes_Ratio'] = input_df['IR'] / input_df['Completes']
        input_df['LOI_Completes_Ratio'] = input_df['LOI'] / input_df['Completes']
        
        # Advanced features
        input_df['IR_LOI_Product'] = input_df['IR'] * input_df['LOI']  # Interaction term
        input_df['CPI_per_Minute'] = 0  # Placeholder since we don't know CPI yet
        input_df['Log_Completes'] = np.log1p(input_df['Completes'])  # Log transformation
        
        # Add Type columns (one-hot encoded)
        input_df['Type_Won'] = 1  # Assuming we want to predict for 'Won' type
        
        # Ensure the input DataFrame has all required columns in the right order
        final_input = pd.DataFrame(columns=feature_names)
        for col in feature_names:
            if col in input_df.columns:
                final_input[col] = input_df[col]
            else:
                final_input[col] = 0
                logger.warning(f"Feature {col} not found in input data, using default value 0")
        
        # Make predictions with each model
        predictions = {}
        for name, model in models.items():
            try:
                pred = model.predict(final_input)[0]
                predictions[name] = pred
                logger.info(f"{name} prediction: ${pred:.2f}")
            except Exception as e:
                logger.error(f"Error making prediction with {name} model: {e}")
                predictions[name] = None
        
        # Filter out None values
        predictions = {k: v for k, v in predictions.items() if v is not None}
        
        return predictions
    
    except Exception as e:
        logger.error(f"Error in predict_cpi: {e}", exc_info=True)
        return {}

def get_prediction_metrics(predictions: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate summary metrics for multiple model predictions.
    
    Args:
        predictions (Dict[str, float]): Dictionary of model predictions
    
    Returns:
        Dict[str, float]: Dictionary of prediction metrics
    """
    try:
        # Check if predictions dictionary is empty
        if not predictions:
            return {}
        
        # Extract prediction values
        values = list(predictions.values())
        
        # Calculate metrics
        metrics = {
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values),
            'median': sorted(values)[len(values) // 2],
            'range': max(values) - min(values),
            'std': np.std(values)
        }
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error in get_prediction_metrics: {e}", exc_info=True)
        return {}

def get_recommendation(predicted_cpi: float, won_avg: float, lost_avg: float) -> str:
    """
    Generate a pricing recommendation based on predictions.
    
    Args:
        predicted_cpi (float): Predicted CPI value
        won_avg (float): Average CPI for won bids
        lost_avg (float): Average CPI for lost bids
    
    Returns:
        str: Recommendation text
    """
    try:
        midpoint = (won_avg + lost_avg) / 2
        diff_percentage = ((predicted_cpi - won_avg) / won_avg) * 100
        
        # Define recommendation based on where the prediction falls
        if predicted_cpi <= won_avg * 0.9:
            recommendation = (
                "This CPI is significantly lower than the average for won bids. While this will "
                "increase chances of winning, it may be unnecessarily low and could reduce profitability. "
                f"Consider raising the price closer to the average won bid of ${won_avg:.2f}."
            )
        elif predicted_cpi <= won_avg:
            recommendation = (
                "This CPI is lower than the average for won bids, suggesting a very competitive "
                "price point that should increase chances of winning while maintaining good profitability."
            )
        elif predicted_cpi <= midpoint:
            recommendation = (
                "This CPI is higher than the average for won bids but still below the midpoint between "
                "won and lost bids, suggesting a moderately competitive price point. It offers a good "
                "balance between win probability and profitability."
            )
        elif predicted_cpi <= lost_avg:
            recommendation = (
                "This CPI is in the upper range between won and lost bids, which may reduce chances of "
                "winning but could improve profitability if the bid is accepted. Consider whether there "
                "are other factors that might justify this premium pricing."
            )
        else:
            recommendation = (
                "This CPI is higher than the average for lost bids, suggesting a price point that may "
                "be too high to be competitive. Unless there are unique selling points or special "
                f"requirements, consider reducing the price to below ${lost_avg:.2f}."
            )
        
        # Add percentage comparison
        recommendation += f" (The predicted CPI is {diff_percentage:+.1f}% compared to the average won bid price.)"
        
        return recommendation
    
    except Exception as e:
        logger.error(f"Error in get_recommendation: {e}", exc_info=True)
        return "Unable to generate recommendation due to an error."

def get_detailed_pricing_strategy(predicted_cpi: float, user_input: Dict[str, float],
                               won_data: pd.DataFrame, lost_data: pd.DataFrame) -> str:
    """
    Generate a detailed pricing strategy based on predictions and user input.
    
    Args:
        predicted_cpi (float): Predicted CPI value
        user_input (Dict[str, float]): Dictionary of user input values
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
    
    Returns:
        str: Detailed pricing strategy text
    """
    try:
        # Extract user input parameters
        ir = user_input['IR']
        loi = user_input['LOI']
        completes = user_input['Completes']
        
        # Filter data by similar parameters
        ir_lower = max(0, ir - 10)
        ir_upper = min(100, ir + 10)
        loi_lower = max(0, loi - 5)
        loi_upper = loi + 5
        completes_lower = max(0, completes * 0.5)
        completes_upper = completes * 1.5
        
        # Filter won data
        similar_won = won_data[
            (won_data['IR'] >= ir_lower) & (won_data['IR'] <= ir_upper) &
            (won_data['LOI'] >= loi_lower) & (won_data['LOI'] <= loi_upper) &
            (won_data['Completes'] >= completes_lower) & (won_data['Completes'] <= completes_upper)
        ]
        
        # Filter lost data
        similar_lost = lost_data[
            (lost_data['IR'] >= ir_lower) & (lost_data['IR'] <= ir_upper) &
            (lost_data['LOI'] >= loi_lower) & (lost_data['LOI'] <= loi_upper) &
            (lost_data['Completes'] >= completes_lower) & (lost_data['Completes'] <= completes_upper)
        ]
        
        # Build detailed strategy text
        strategy = []
        
        # Add heading
        strategy.append(f"### Detailed Pricing Strategy for IR={ir}%, LOI={loi} min, Completes={completes}")
        strategy.append("")
        
        # Add comparison with similar projects
        if len(similar_won) > 0:
            strategy.append(f"**Similar Won Projects:** {len(similar_won)}")
            strategy.append(f"- Average CPI: ${similar_won['CPI'].mean():.2f}")
            strategy.append(f"- Min CPI: ${similar_won['CPI'].min():.2f}")
            strategy.append(f"- Max CPI: ${similar_won['CPI'].max():.2f}")
            strategy.append("")
        
        if len(similar_lost) > 0:
            strategy.append(f"**Similar Lost Projects:** {len(similar_lost)}")
            strategy.append(f"- Average CPI: ${similar_lost['CPI'].mean():.2f}")
            strategy.append(f"- Min CPI: ${similar_lost['CPI'].min():.2f}")
            strategy.append(f"- Max CPI: ${similar_lost['CPI'].max():.2f}")
            strategy.append("")
        
        # Add specific recommendations
        strategy.append("**Recommended Pricing Adjustments:**")
        
        # IR-based adjustment
        if ir < 20:
            strategy.append("- **Low IR Adjustment:** Add a premium of 15-20% to the base CPI to account for the difficulty of finding qualified respondents.")
        elif ir < 50:
            strategy.append("- **Medium IR Adjustment:** Add a premium of 5-10% to the base CPI.")
        else:
            strategy.append("- **High IR Adjustment:** No IR premium needed as the incidence rate is high.")
        
        # LOI-based adjustment
        if loi > 20:
            strategy.append("- **Long LOI Adjustment:** Add a premium of 15-20% to compensate for the longer survey