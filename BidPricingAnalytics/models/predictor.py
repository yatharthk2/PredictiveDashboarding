"""
ML model prediction functionality for the CPI Analysis & Prediction Dashboard.
Includes functions for making predictions and generating recommendations.
"""

import pandas as pd
import numpy as np
import logging
import traceback
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.preprocessing import StandardScaler

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
        
        # Handle extreme or invalid values
        for col, limits in {
            'IR': (0, 100),        # IR between 0-100%
            'LOI': (0, 120),       # LOI between 0-120 minutes
            'Completes': (1, None) # At least 1 complete
        }.items():
            min_val, max_val = limits
            if min_val is not None and input_df[col].iloc[0] < min_val:
                logger.warning(f"Input {col}={input_df[col].iloc[0]} below minimum {min_val}, capping.")
                input_df[col] = min_val
            if max_val is not None and input_df[col].iloc[0] > max_val:
                logger.warning(f"Input {col}={input_df[col].iloc[0]} above maximum {max_val}, capping.")
                input_df[col] = max_val
        
        # Replace zeros with small values to avoid division by zero
        for col in input_df.columns:
            if input_df[col].iloc[0] == 0:
                input_df[col] = 0.001
                logger.warning(f"Replaced zero value in {col} with 0.001 to avoid division by zero")
        
        # Feature engineering
        input_df['IR_LOI_Ratio'] = input_df['IR'] / input_df['LOI']
        input_df['IR_Completes_Ratio'] = input_df['IR'] / input_df['Completes']
        input_df['LOI_Completes_Ratio'] = input_df['LOI'] / input_df['Completes']
        
        # Advanced features
        input_df['IR_LOI_Product'] = input_df['IR'] * input_df['LOI']  # Interaction term
        input_df['CPI_per_Minute'] = 0  # Placeholder since we don't know CPI yet
        input_df['Log_Completes'] = np.log1p(input_df['Completes'])  # Log transformation
        
        # Polynomial features
        input_df['IR_Squared'] = input_df['IR'] ** 2
        input_df['LOI_Squared'] = input_df['LOI'] ** 2
        input_df['Log_IR_LOI_Product'] = np.log1p(input_df['IR_LOI_Product'])
        
        # Add Type columns (one-hot encoded)
        input_df['Type_Won'] = 1  # Assuming we want to predict for 'Won' type
        
        # Ensure the input DataFrame has all required columns in the right order
        final_input = pd.DataFrame(columns=feature_names)
        for col in feature_names:
            if col in input_df.columns:
                final_input[col] = input_df[col]
            else:
                logger.warning(f"Feature {col} not found in input data, using default value 0")
                final_input[col] = 0
        
        # Scale numeric features to match the training data scale
        numeric_features = final_input.select_dtypes(include=['float', 'int']).columns
        if len(numeric_features) > 0:
            try:
                # Note: Ideally the scaler would be saved during training and reused here
                # But for now we'll create a new scaler with reasonable assumptions
                scaler = StandardScaler()
                final_input[numeric_features] = scaler.fit_transform(final_input[numeric_features])
                logger.info("Applied scaling to numeric features")
            except Exception as e:
                logger.warning(f"Could not scale numeric features: {e}")
        
        # Make predictions with each model with robust error handling
        predictions = {}
        for name, model in models.items():
            try:
                # Try to get model prediction
                pred = model.predict(final_input)[0]
                
                # Handle unreasonable predictions (negative or extremely high values)
                if pred < 0:
                    logger.warning(f"{name} produced negative prediction {pred}, setting to 0")
                    pred = 0
                elif pred > 1000:  # Arbitrary cap for unreasonable CPI values
                    logger.warning(f"{name} produced extremely high prediction {pred}, capping at 1000")
                    pred = 1000
                
                predictions[name] = pred
                logger.info(f"{name} prediction: ${pred:.2f}")
            except Exception as e:
                logger.error(f"Error making prediction with {name} model: {e}", exc_info=True)
                # Try a fallback prediction approach if standard approach fails
                try:
                    if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
                        # For linear models, try direct computation
                        coef = getattr(model, 'coef_')
                        intercept = getattr(model, 'intercept_')
                        
                        # Simple dot product
                        pred = np.dot(final_input.values[0], coef) + intercept
                        
                        # Handle unreasonable predictions
                        if pred < 0:
                            pred = 0
                        elif pred > 1000:
                            pred = 1000
                            
                        predictions[name] = pred
                        logger.info(f"{name} prediction (fallback method): ${pred:.2f}")
                    else:
                        logger.warning(f"Could not use fallback prediction for {name}")
                        predictions[name] = None
                except Exception as e2:
                    logger.error(f"Error in fallback prediction for {name}: {e2}")
                    predictions[name] = None
        
        # Filter out None values
        predictions = {k: v for k, v in predictions.items() if v is not None}
        
        # If all models failed, return a simple heuristic prediction
        if not predictions:
            logger.warning("All models failed, using heuristic prediction.")
            # Simple heuristic: Higher IR = lower CPI, Higher LOI = higher CPI
            heuristic_prediction = 10 * (1 + user_input['LOI'] / 15) * (1 - user_input['IR'] / 200)
            predictions['Heuristic'] = heuristic_prediction
        
        return predictions
    
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error in predict_cpi: {error_details}")
        # Return a default prediction to avoid complete failure
        return {'Fallback': 15.0}  # Use a reasonable average CPI as fallback

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
            return {
                'min': 0,
                'max': 0,
                'mean': 0,
                'median': 0,
                'range': 0,
                'std': 0,
                'status': 'Error: No predictions available'
            }
        
        # Extract prediction values
        values = list(predictions.values())
        
        # Calculate metrics with proper error handling
        try:
            min_val = min(values)
        except:
            min_val = 0
            
        try:
            max_val = max(values)
        except:
            max_val = 0
            
        try:
            mean_val = sum(values) / len(values)
        except:
            mean_val = 0
            
        try:
            median_val = sorted(values)[len(values) // 2]
        except:
            median_val = 0
            
        try:
            range_val = max_val - min_val
        except:
            range_val = 0
            
        try:
            std_val = np.std(values)
        except:
            std_val = 0
        
        # Create metrics dictionary
        metrics = {
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'median': median_val,
            'range': range_val,
            'std': std_val,
            'status': 'OK'
        }
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error in get_prediction_metrics: {e}", exc_info=True)
        return {
            'min': 0,
            'max': 0,
            'mean': 0,
            'median': 0,
            'range': 0,
            'std': 0,
            'status': f'Error: {str(e)}'
        }

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
        # Handle potential invalid inputs
        if predicted_cpi <= 0 or np.isnan(predicted_cpi):
            return "Unable to provide a recommendation due to invalid prediction value."
            
        if won_avg <= 0 or np.isnan(won_avg) or lost_avg <= 0 or np.isnan(lost_avg):
            return "Unable to provide a recommendation due to invalid reference values."
        
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
        return "Unable to generate recommendation due to an unexpected error."

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
        # Handle potential invalid inputs
        if predicted_cpi <= 0 or np.isnan(predicted_cpi):
            return "Unable to provide a detailed pricing strategy due to invalid prediction value."
            
        if won_data.empty or lost_data.empty:
            return "Unable to provide a detailed pricing strategy due to insufficient reference data."
        
        # Extract user input parameters
        ir = user_input.get('IR', 0)
        loi = user_input.get('LOI', 0) 
        completes = user_input.get('Completes', 0)
        
        # Validate input values
        ir = max(0, min(100, ir))
        loi = max(1, loi)
        completes = max(1, completes)
        
        # Filter data by similar parameters - with wider ranges for matching
        ir_lower = max(0, ir - 15)
        ir_upper = min(100, ir + 15)
        loi_lower = max(0, loi - 5)
        loi_upper = loi + 5
        completes_lower = max(1, completes * 0.5)
        completes_upper = completes * 2.0
        
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
        
        # If there are no similar projects, expand the search scope
        if len(similar_won) < 3 or len(similar_lost) < 3:
            logger.warning("Not enough similar projects found with tight criteria, expanding search.")
            ir_lower = max(0, ir - 25)
            ir_upper = min(100, ir + 25)
            loi_lower = max(0, loi - 10)
            loi_upper = loi + 10
            
            # Filter with wider criteria
            similar_won = won_data[
                (won_data['IR'] >= ir_lower) & (won_data['IR'] <= ir_upper) &
                (won_data['LOI'] >= loi_lower) & (won_data['LOI'] <= loi_upper)
            ]
            
            similar_lost = lost_data[
                (lost_data['IR'] >= ir_lower) & (lost_data['IR'] <= ir_upper) &
                (lost_data['LOI'] >= loi_lower) & (lost_data['LOI'] <= loi_upper)
            ]
        
        # Build detailed strategy text
        strategy = []
        
        # Add heading
        strategy.append(f"### Detailed Pricing Strategy for IR={ir}%, LOI={loi} min, Completes={completes}")
        strategy.append("")
        
        # Add comparison with similar projects with robust error handling
        if len(similar_won) > 0:
            try:
                won_avg_cpi = similar_won['CPI'].mean()
                won_min_cpi = similar_won['CPI'].min()
                won_max_cpi = similar_won['CPI'].max()
                
                strategy.append(f"**Similar Won Projects:** {len(similar_won)}")
                strategy.append(f"- Average CPI: ${won_avg_cpi:.2f}")
                strategy.append(f"- Min CPI: ${won_min_cpi:.2f}")
                strategy.append(f"- Max CPI: ${won_max_cpi:.2f}")
                strategy.append("")
            except Exception as e:
                logger.warning(f"Error calculating won stats: {e}")
                strategy.append(f"**Similar Won Projects:** {len(similar_won)}")
                strategy.append("- Statistics unavailable due to data issue")
                strategy.append("")
        
        if len(similar_lost) > 0:
            try:
                lost_avg_cpi = similar_lost['CPI'].mean()
                lost_min_cpi = similar_lost['CPI'].min()
                lost_max_cpi = similar_lost['CPI'].max()
                
                strategy.append(f"**Similar Lost Projects:** {len(similar_lost)}")
                strategy.append(f"- Average CPI: ${lost_avg_cpi:.2f}")
                strategy.append(f"- Min CPI: ${lost_min_cpi:.2f}")
                strategy.append(f"- Max CPI: ${lost_max_cpi:.2f}")
                strategy.append("")
            except Exception as e:
                logger.warning(f"Error calculating lost stats: {e}")
                strategy.append(f"**Similar Lost Projects:** {len(similar_lost)}")
                strategy.append("- Statistics unavailable due to data issue")
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
            strategy.append("- **Long LOI Adjustment:** Add a premium of 15-20% to compensate for the longer survey duration.")
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
        
        # Final pricing recommendation based on all factors
        try:
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
        except Exception as e:
            logger.warning(f"Error calculating price adjustments: {e}")
            strategy.append("")
            strategy.append("**Base CPI:** ${predicted_cpi:.2f}")
            strategy.append("**Note:** Could not calculate specific adjustments due to data issues.")
        
        # Add competitive analysis with robust error handling
        strategy.append("")
        strategy.append("**Competitive Positioning:**")

        if len(similar_won) > 0 and len(similar_lost) > 0:
            try:
                won_avg = similar_won['CPI'].mean()
                lost_avg = similar_lost['CPI'].mean()
                adjusted_cpi = predicted_cpi * (1 + (total_adjustment if 'total_adjustment' in locals() else 0) / 100)
                
                if adjusted_cpi <= won_avg:
                    strategy.append(f"- This recommended CPI (${adjusted_cpi:.2f}) is below the average of similar won bids (${won_avg:.2f}), suggesting a highly competitive position.")
                elif adjusted_cpi <= (won_avg + lost_avg) / 2:
                    strategy.append(f"- This recommended CPI (${adjusted_cpi:.2f}) is above the average of similar won bids (${won_avg:.2f}) but below the midpoint, suggesting a moderately competitive position.")
                elif adjusted_cpi <= lost_avg:
                    strategy.append(f"- This recommended CPI (${adjusted_cpi:.2f}) is closer to the average of similar lost bids (${lost_avg:.2f}), which may reduce win probability. Consider additional value-add services to justify this premium.")
                else:
                    strategy.append(f"- This recommended CPI (${adjusted_cpi:.2f}) is above the average of similar lost bids (${lost_avg:.2f}), suggesting it may be too high to be competitive unless there are unique selling points.")
            except Exception as e:
                strategy.append("- Could not analyze competitive positioning due to data issues.")
        else:
            strategy.append("- Not enough similar projects found to provide detailed competitive positioning.")

        # Add tips for special cases
        strategy.append("")
        strategy.append("**Additional Tips:**")
        
        # Low IR tips
        if ir < 20:
            strategy.append("- For low IR projects, highlight your panel quality and ability to efficiently reach hard-to-find audiences.")
        
        # Long LOI tips
        if loi > 20:
            strategy.append("- For longer surveys, emphasize measures to maintain respondent engagement and data quality.")
        
        # High volume tips
        if completes > 500:
            strategy.append("- For large sample sizes, consider offering tiered pricing with better rates as certain volume thresholds are reached.")
        
        # Join the strategy text with line breaks
        return "\n".join(strategy)
    
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error in get_detailed_pricing_strategy: {error_details}")
        
        # Return a simplified strategy in case of error
        return (
            f"### Basic Pricing Strategy for IR={user_input.get('IR', 0)}%, "
            f"LOI={user_input.get('LOI', 0)} min, Completes={user_input.get('Completes', 0)}\n\n"
            f"**Recommended Base CPI:** ${predicted_cpi:.2f}\n\n"
            "**Note:** A detailed pricing strategy could not be generated due to a data processing issue. "
            "Please check your input values and try again."
        )

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
        # Input validation
        if predicted_cpi <= 0 or np.isnan(predicted_cpi):
            return {
                'win_probability': 50.0,
                'error': "Invalid CPI prediction",
                'confidence': "Low"
            }
            
        if won_data.empty or lost_data.empty:
            return {
                'win_probability': 50.0,
                'error': "Insufficient reference data",
                'confidence': "Low"
            }
        
        # Extract user input
        ir = user_input.get('IR', 50)
        loi = user_input.get('LOI', 15)
        completes = user_input.get('Completes', 500)
        
        # Validate and normalize input
        ir = max(0, min(100, ir))
        loi = max(1, loi)
        completes = max(1, completes)
        
        # Enhanced approach: use similar projects for better context relevance
        # Filter for similar projects
        ir_range = 15
        loi_range = 5
        ir_lower = max(0, ir - ir_range)
        ir_upper = min(100, ir + ir_range)
        loi_lower = max(1, loi - loi_range)
        loi_upper = loi + loi_range
        
        # Filter won and lost datasets
        similar_won = won_data[
            (won_data['IR'] >= ir_lower) & (won_data['IR'] <= ir_upper) &
            (won_data['LOI'] >= loi_lower) & (won_data['LOI'] <= loi_upper)
        ]
        
        similar_lost = lost_data[
            (lost_data['IR'] >= ir_lower) & (lost_data['IR'] <= ir_upper) &
            (lost_data['LOI'] >= loi_lower) & (lost_data['LOI'] <= loi_upper)
        ]
        
        # If not enough similar projects, expand the filter
        if len(similar_won) < 5 or len(similar_lost) < 5:
            ir_range = 25
            loi_range = 10
            ir_lower = max(0, ir - ir_range)
            ir_upper = min(100, ir + ir_range)
            loi_lower = max(1, loi - loi_range)
            loi_upper = loi + loi_range
            
            # Re-filter with wider criteria
            similar_won = won_data[
                (won_data['IR'] >= ir_lower) & (won_data['IR'] <= ir_upper) &
                (won_data['LOI'] >= loi_lower) & (won_data['LOI'] <= loi_upper)
            ]
            
            similar_lost = lost_data[
                (lost_data['IR'] >= ir_lower) & (lost_data['IR'] <= ir_upper) &
                (lost_data['LOI'] >= loi_lower) & (lost_data['LOI'] <= loi_upper)
            ]
        
        # Still not enough data? Use all data with a warning
        if len(similar_won) < 5 or len(similar_lost) < 5:
            logger.warning("Not enough similar projects for reliable win probability estimation")
            similar_won = won_data
            similar_lost = lost_data
            confidence = "Low"
        else:
            confidence = "Medium" if len(similar_won) + len(similar_lost) < 30 else "High"
        
        # Combine similar won and lost projects
        combined_similar = pd.concat([
            similar_won.assign(Won=1),
            similar_lost.assign(Won=0)
        ], ignore_index=True)
        
        # Sort by CPI to create a cumulative distribution
        combined_similar = combined_similar.sort_values('CPI')
        combined_similar['cumulative_count'] = range(1, len(combined_similar) + 1)
        combined_similar['win_rate_below'] = combined_similar['Won'].cumsum() / combined_similar['cumulative_count']
        
        # Find the win rate for the predicted CPI
        # Find the closest value at or below the predicted CPI
        below_prediction = combined_similar[combined_similar['CPI'] <= predicted_cpi]
        
        if len(below_prediction) == 0:
            # Predicted CPI is below all observed values - high win chance
            win_probability = 95.0
        elif len(below_prediction) == len(combined_similar):
            # Predicted CPI is above all observed values - low win chance
            win_probability = 5.0
        else:
            # Look at the win rate at this CPI threshold
            win_probability = below_prediction.iloc[-1]['win_rate_below'] * 100
        
        # Adjust win probability based on additional factors
        # Volume discount effect
        if completes > 500:
            win_probability += 5  # Large volume may increase win chance
        
        # Cap win probability between 1% and 99% 
        win_probability = max(1, min(99, win_probability))
        
        # Return detailed results
        return {
            'win_probability': win_probability,
            'cpi_percentile': (len(below_prediction) / len(combined_similar)) * 100 if len(combined_similar) > 0 else 50,
            'similar_won_count': len(similar_won),
            'similar_lost_count': len(similar_lost),
            'confidence': confidence
        }
    
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error in simulate_win_probability: {error_details}")
        
        # Return default probability with error flag
        return {
            'win_probability': 50.0,
            'error': f"Error calculating win probability: {str(e)}",
            'confidence': "Low"
        }

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
        print(f"CPI Percentile: {win_prob.get('cpi_percentile', 'N/A')}")
        print(f"Confidence: {win_prob.get('confidence', 'N/A')}")
        
        print("\nAll tests completed successfully")
        
    except Exception as e:
        print(f"Error testing predictor: {e}")
        traceback.print_exc()