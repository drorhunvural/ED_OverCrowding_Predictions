#!/usr/bin/env python
# predict_boarding_count_v2.py - Predict future boarding count using trained model

import numpy as np
import pandas as pd
import os
import joblib
import torch
import warnings
import platform
import sys

# Suppress warnings
warnings.filterwarnings("ignore")

# Define the custom metrics function that was used during training
def mean_squared_error_fastai(y_pred, y_true):
    y_pred, y_true = y_pred.detach().cpu().numpy(), y_true.detach().cpu().numpy()
    if np.isnan(y_pred).any() or np.isnan(y_true).any():
        raise ValueError("NaN encountered in predictions or targets.")
    return ((y_true - y_pred) ** 2).mean()

# Add to __main__ namespace for model loading
import __main__
__main__.mean_squared_error_fastai = mean_squared_error_fastai

# Import tsai functions
try:
    from tsai.inference import load_learner
except ImportError:
    from tsai.basics import load_learner

def predict_future_boarding_count(input_data, feature_names):
    """
    Predict future boarding count using trained model
    
    Args:
        input_data: List of input features in the correct order
        feature_names: List of feature names corresponding to input_data
        
    Returns:
        Predicted future boarding count
    """
    # Setup paths
    base_path = "boarding_count_tsai"
    model_architecture = "TSTPlus"
    dataset_name = "dataset3"
    model_file = "reg_TSTPlus_dataset3_bs64_dr0.103519_wd0.091335_lr0.031357_optLamb_mae_4.3039.pkl"
    
    model_path = os.path.join(base_path, "models", model_architecture, model_file)
    target_scaler_path = os.path.join(base_path, "datasets", dataset_name, f"target_scaler_{dataset_name}.pkl")
    feature_scaler_path = os.path.join(base_path, "datasets", dataset_name, f"feature_scaler_{dataset_name}.pkl")
    
    # Check paths exist
    for path in [model_path, target_scaler_path, feature_scaler_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")
    
    # Load feature scaler
    feature_scaler = joblib.load(feature_scaler_path)
    
    # Create DataFrame with feature names
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Get scaled features list from the scaler
    if hasattr(feature_scaler, 'feature_names_in_'):
        scaled_features = list(feature_scaler.feature_names_in_)
        # print(f"\nFeatures expected by scaler ({len(scaled_features)}):")
        # print(", ".join(scaled_features))
        
        # Select only the columns that the scaler expects
        scaling_df = input_df[scaled_features].copy()
        
        # Apply scaling to those columns
        scaled_values = feature_scaler.transform(scaling_df)
        
        # Create a new DataFrame with all original columns
        scaled_df = input_df.copy()
        
        # Update the scaled columns
        for i, col in enumerate(scaled_features):
            scaled_df[col] = scaled_values[0, i]
        
        # Get the final numpy array of all features (scaled and unscaled)
        input_scaled = scaled_df.values
    else:
        # If scaler doesn't have feature names, try a more generic approach
        print("\nScaler doesn't have feature names. Using a more generic approach.")
        # Convert input to numpy array and reshape for scaling
        input_array = np.array(input_data, dtype=np.float32).reshape(1, -1)
        
        # Try to apply scaling to the subset of features the scaler expects
        if hasattr(feature_scaler, 'n_features_in_'):
            n_features = feature_scaler.n_features_in_
            # print(f"Scaler expects {n_features} features out of {len(input_data)} total features.")
            
            # Assume the first n_features are the ones to scale
            scaling_subset = input_array[:, :n_features]
            scaled_subset = feature_scaler.transform(scaling_subset)
            
            # Combine scaled subset with unscaled features
            input_scaled = input_array.copy()
            input_scaled[:, :n_features] = scaled_subset
        else:
            # If we can't determine feature count, just try direct scaling
            print("WARNING: Cannot determine feature count. Attempting direct scaling.")
            try:
                input_scaled = feature_scaler.transform(input_array)
            except Exception as e:
                raise ValueError(f"Failed to scale features: {str(e)}")
    
    # Add sequence dimension for time series model
    input_3d = input_scaled.reshape(1, input_scaled.shape[1], 1)
    print(f"Input shape after preprocessing: {input_3d.shape}")
    
    # Load model - no Windows patch needed, load directly
    # print(f"Loading model from {model_path}")
    model = load_learner(model_path, cpu=True)
    
    # Put model in evaluation mode
    if hasattr(model, 'model'):
        model.model.eval()
    
    # Make prediction
    print("Making prediction...")
    with torch.no_grad():
        input_tensor = torch.tensor(input_3d, dtype=torch.float32)
        prediction = model.model(input_tensor)
        
    # Convert prediction to numpy
    pred_numpy = prediction.cpu().numpy()
    
    # Load target scaler and inverse transform
    # print(f"Loading target scaler from {target_scaler_path}")
    target_scaler = joblib.load(target_scaler_path)
    pred_original = target_scaler.inverse_transform(pred_numpy.reshape(-1, 1))
    
    # Return single prediction value
    return float(pred_original[0, 0])

if __name__ == "__main__":
    # Define feature names for clarity
    feature_names = [
        "UED_Hospital_Census", "total_patient_count", "cumulative_total_average_waiting_time", 
        "temp", "boarding_count", "average_boarding_times_minute", "treatment_count", 
        "average_waiting_times_treatment_minute", "Alabama_Football_Game_Actual_Time", 
        "Federal_Holiday", "Auburn_Football_Game_Actual_Time", "year", "month", 
        "day_of_month", "hour", "day_of_week", "weather_Clear", "weather_Clouds", 
        "weather_Others", "weather_Rain", "weather_Thunderstorm", "extreme_indicator",
        "lag_1_boarding_count", "lag_2_boarding_count", "lag_3_boarding_count", 
        "lag_4_boarding_count", "lag_5_boarding_count", "lag_6_boarding_count", 
        "lag_7_boarding_count", "lag_8_boarding_count", "lag_9_boarding_count", 
        "lag_10_boarding_count", "lag_11_boarding_count", "lag_12_boarding_count"
    ]
    
    # Example 1: High boarding count (Real output ≈ 57)
    example1 = [
        819, 29, 179, 292, 56, 906, 75, 784,   # Hospital metrics
        0, 0, 0,                               # Special events
        2022, 11, 10, 23, 3,                   # Date/time
        0, 0, 0, 1, 0, 1,                      # Weather and extreme flag
        57, 55, 57, 55, 46, 49, 52, 49, 49, 48, 48, 50  # Historical boarding counts
    ] # Real Output is 57
    
    # Example 2: Holiday with low boarding count (Real output ≈ 5)
    example2 = [
        851, 28, 112, 284, 40, 839, 59, 687,  # Hospital metrics
        0, 0, 0,                              # Special events
        2022, 11, 18, 15, 4,                  # Date/time
        0, 1, 0, 0, 0, 1,                     # Weather and extreme flag
        41, 39, 39, 41, 42, 43, 42, 44, 43, 45, 49, 48  # Historical boarding counts
    ] #Real Output is 29
    
    # Example 3: Moderate boarding count (Real output ≈ 43)
    example3 = [
        719, 5, 5, 266, 7, 106, 26, 258,     # Hospital metrics
        0, 1, 0,                             # Special events
        2022, 12, 26, 6, 0,                  # Date/time
        1, 0, 0, 0, 0, 0,                    # Weather and extreme flag
        5, 7, 7, 10, 15, 17, 11, 14, 13, 12, 12, 17  # Historical boarding counts
    ] #Real Output is 9
    
    # Select which example to use
    current_example = example3
 
    

    print("=" * 60)
    print(f"{'Feature':<40} {'Value'}")
    print("-" * 60)
    for name, value in zip(feature_names, current_example):
        print(f"{name:<40} {value}")
    print("=" * 60)
    
    try:
        # Print system info for debugging
        # print(f"System: {platform.system()}")
        # print(f"Python version: {sys.version}")
        
        # Make prediction
        prediction = predict_future_boarding_count(current_example, feature_names)
        
        # Print results
        print("\nRESULTS:")
        print(f"Predicted Future Boarding Count: {prediction:.2f}")

        
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        import traceback
        traceback.print_exc()


#python predict_tsai.py