#!/usr/bin/env python
# predict.py - Generic prediction script for trained tsai models

import numpy as np
import pandas as pd
import os
import joblib
import torch
import warnings
import json
import sys
from pathlib import Path

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

def load_dataset_info(dataset_path):
    """Load dataset configuration and feature information"""
    config_path = os.path.join(dataset_path, f"config_{os.path.basename(dataset_path)}.json")
    explanation_path = os.path.join(dataset_path, f"explanation_{os.path.basename(dataset_path)}.txt")
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Parse explanation file for feature order
    with open(explanation_path, 'r') as f:
        explanation = f.read()
    
    # Extract feature order from explanation
    feature_order_start = explanation.find("Feature order used (for model input):")
    if feature_order_start == -1:
        raise ValueError("Could not find feature order in explanation file")
    
    feature_order_line = explanation[feature_order_start:].split('\n')[1]
    # Parse the list string
    feature_order = eval(feature_order_line)
    
    # Extract scaled features
    scaled_features_start = explanation.find("Features scaled:")
    if scaled_features_start != -1:
        scaled_features_line = explanation[scaled_features_start:].split('\n')[0]
        scaled_features_str = scaled_features_line.split(": ")[1]
        scaled_features = eval(scaled_features_str)
    else:
        scaled_features = []
    
    return config, feature_order, scaled_features

def predict_with_model(input_data, model_path, dataset_path, feature_names=None):
    """
    Make prediction using trained model
    
    Args:
        input_data: List/array of input features in the correct order
        model_path: Path to the trained .pkl model file
        dataset_path: Path to the dataset folder containing scalers and config
        feature_names: Optional list of feature names (for validation)
        
    Returns:
        Predicted value (inverse scaled)
    """
    
    # Load dataset info
    config, feature_order, scaled_features = load_dataset_info(dataset_path)
    dataset_name = os.path.basename(dataset_path)
    
    # Validate input data length
    if len(input_data) != len(feature_order):
        raise ValueError(f"Input data length ({len(input_data)}) doesn't match expected features ({len(feature_order)})")
    
    # If feature names provided, validate order
    if feature_names:
        if feature_names != feature_order:
            print("WARNING: Feature names don't match expected order!")
            print(f"Expected: {feature_order}")
            print(f"Provided: {feature_names}")
    
    # Convert to numpy array
    input_array = np.array(input_data, dtype=np.float32).reshape(1, -1)
    
    # Create DataFrame for easier handling
    input_df = pd.DataFrame(input_array, columns=feature_order)
    
    # Load and apply feature scaling
    feature_scaler_path = os.path.join(dataset_path, f"feature_scaler_{dataset_name}.pkl")
    if os.path.exists(feature_scaler_path) and scaled_features:
        feature_scaler = joblib.load(feature_scaler_path)
        
        # Scale only the features that were scaled during training
        scaled_df = input_df.copy()
        if scaled_features:
            scaled_values = feature_scaler.transform(input_df[scaled_features])
            scaled_df[scaled_features] = scaled_values
        
        input_scaled = scaled_df.values
    else:
        # No scaling needed
        input_scaled = input_array
    
    # Add sequence dimension for time series model (samples, features, sequence_length)
    input_3d = input_scaled.reshape(1, input_scaled.shape[1], 1)
    
    print(f"Input shape after preprocessing: {input_3d.shape}")
    print(f"Expected features: {len(feature_order)}")
    
    # Load model
    print(f"Loading model from {model_path}")
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
    target_scaler_path = os.path.join(dataset_path, f"target_scaler_{dataset_name}.pkl")
    if os.path.exists(target_scaler_path):
        target_scaler = joblib.load(target_scaler_path)
        pred_original = target_scaler.inverse_transform(pred_numpy.reshape(-1, 1))
        return float(pred_original[0, 0])
    else:
        # No target scaling was applied
        return float(pred_numpy[0, 0])

def main():
    """Main function with example usage"""
    
    # Configuration - UPDATE THESE PATHS
    TRAINED_FOLDER = "trainedfiles"
    DATASETS_FOLDER = "datasets"
    
    # Specify which model and dataset to use
    model_name = "TSTPlus"
    dataset_name = "dataset1"
    model_filename = "reg_TSTPlus_dataset1_trial0_mae112.7824.pkl"  # Update this
    
    # Build paths
    model_path = os.path.join(TRAINED_FOLDER, model_name, dataset_name, model_filename)
    dataset_path = os.path.join(DATASETS_FOLDER, dataset_name)
    
    # Check if paths exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset folder not found: {dataset_path}")
        return
    
    # Load dataset info to see expected features
    try:
        config, feature_order, scaled_features = load_dataset_info(dataset_path)
        print("Dataset Configuration:")
        print(f"  Dataset: {dataset_name}")
        print(f"  Target feature: {config.get('target_feature', 'Unknown')}")
        print(f"  Future hours: {config.get('future_hours', 'Unknown')}")
        print(f"  Number of features: {len(feature_order)}")
        print(f"  Features that are scaled: {len(scaled_features)}")
        print("\nExpected feature order:")
        for i, feature in enumerate(feature_order):
            print(f"  {i+1:2d}. {feature}")
        print()
        
    except Exception as e:
        print(f"Error loading dataset info: {e}")
        return
    
    # Example input data - UPDATE THIS WITH YOUR ACTUAL DATA
    # This should match the feature order exactly
    example_input = [
        1200,    # average_boarding_times_minute
        2023,     # year
        6,        # month
        15,       # day_of_month
        14,       # hour
        3,        # day_of_week (Thursday)
        0,        # extreme_indicator
        1000.0,    # lag_1_average_boarding_times_minute
        980.0,    # lag_2_average_boarding_times_minute
        1400.0,    # lag_3_average_boarding_times_minute
        1600.0,    # lag_4_average_boarding_times_minute
        750.0,    # lag_5_average_boarding_times_minute
        800.0,    # lag_6_average_boarding_times_minute
        820.0,    # lag_7_average_boarding_times_minute
        780.0,    # lag_8_average_boarding_times_minute
        760.0,    # lag_9_average_boarding_times_minute
        740.0,    # lag_10_average_boarding_times_minute
        730.0,    # lag_11_average_boarding_times_minute
        720.0,    # lag_12_average_boarding_times_minute
    ] 
    
    # Validate input length
    if len(example_input) != len(feature_order):
        print(f"Error: Input data has {len(example_input)} values but expected {len(feature_order)}")
        print("Please update the example_input to match the expected features.")
        return
    
    # Display input data
    print("Input Data:")
    print("=" * 80)
    print(f"{'Feature':<40} {'Value'}")
    print("-" * 80)
    for name, value in zip(feature_order, example_input):
        print(f"{name:<40} {value}")
    print("=" * 80)
    
    try:
        # Make prediction
        prediction = predict_with_model(
            input_data=example_input,
            model_path=model_path,
            dataset_path=dataset_path,
            feature_names=feature_order
        )
        
        # Display results
        print("\nRESULTS:")
        print(f"Predicted {config.get('target_feature', 'target')} after {config.get('future_hours', '?')} hours: {prediction:.2f}")
        
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# Usage examples:
# 
# 1. Basic usage with default paths:
#    python predict.py
#
# 2. To use programmatically:
#    from predict import predict_with_model
#    result = predict_with_model(input_data, model_path, dataset_path)
#
# 3. To use with different model:
#    Update model_filename, dataset_name, and example_input in main()