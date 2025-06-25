#!/usr/bin/env python
# timeseries_only_evaluate.py - Evaluation script that only produces the time series plot

import numpy as np
import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import torch
import warnings
import sys

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Import load_learner without printing
try:
    from tsai.inference import load_learner
except ImportError:
    from tsai.basics import load_learner

# Define the custom metrics function that was used in training
def mean_squared_error_fastai(y_pred, y_true):
    y_pred, y_true = y_pred.detach().cpu().numpy(), y_true.detach().cpu().numpy()
    if np.isnan(y_pred).any() or np.isnan(y_true).any():
        raise ValueError("NaN encountered in predictions or targets.")
    return ((y_true - y_pred) ** 2).mean()

def direct_inference(model, x_batch):
    """Run inference directly with the model"""
    if hasattr(model, 'model'):
        pytorch_model = model.model
    else:
        pytorch_model = model
    
    # Put model in eval mode
    pytorch_model.eval()
    
    # Convert to tensor if it's not already
    if not isinstance(x_batch, torch.Tensor):
        x_tensor = torch.tensor(x_batch, dtype=torch.float32)
    else:
        x_tensor = x_batch
        
    # Run inference
    with torch.no_grad():
        pred = pytorch_model(x_tensor)
        return pred
    
    return None

def evaluate_model(model_architecture, dataset_name, model_filename):
    """Evaluate a trained model and generate only the time series plot"""
    # Define paths
    base_path = "boarding_count_tsai"
    models_path = os.path.join(base_path, "models", model_architecture)
    dataset_folder = os.path.join(base_path, "dataset15", dataset_name)
    model_path = os.path.join(models_path, model_filename)
    output_folder = os.path.join(base_path, "results", model_architecture, dataset_name)
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Evaluating {model_architecture} model: {model_filename}")
    
    # Load test data
    X_test_path = os.path.join(dataset_folder, f"X_test_{dataset_name}.npy")
    y_test_path = os.path.join(dataset_folder, f"y_test_{dataset_name}.npy")
    
    print(X_test_path)
    if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        print(f"Error: Test data not found")
        return
    
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    
    # Load target scaler
    target_scaler_path = os.path.join(dataset_folder, f"target_scaler_{dataset_name}.pkl")
    if os.path.exists(target_scaler_path):
        target_scaler = joblib.load(target_scaler_path)
    else:
        print(f"Warning: Target scaler not found")
        target_scaler = None
    
    try:
        # Load the model
        reg = load_learner(model_path, cpu=True)
        
        # Get predictions batch by batch
        all_preds = []
        batch_size = 32
        
        for i in range(0, len(X_test), batch_size):
            end_idx = min(i + batch_size, len(X_test))
            batch_features = X_test[i:end_idx]
            
            # Get predictions
            batch_preds = direct_inference(reg, batch_features)
            
            # Convert to numpy and store
            batch_preds_np = batch_preds.cpu().numpy()
            all_preds.extend(batch_preds_np)
        
        # Convert list of predictions to numpy array
        preds = np.array(all_preds)
        
        # Reshape if needed
        if len(preds.shape) > 1 and preds.shape[1] == 1:
            preds = preds.flatten()
            
        # Ensure y_test is in the right shape
        if len(y_test.shape) > 1 and y_test.shape[1] == 1:
            y_test = y_test.flatten()
        
        # Verify predictions length
        if len(preds) != len(y_test):
            min_len = min(len(preds), len(y_test))
            preds = preds[:min_len]
            y_test = y_test[:min_len]
        
        # Inverse transform if scaler is available
        if target_scaler:
            y_true_original = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_original = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        else:
            y_true_original = y_test
            y_pred_original = preds
        
        # Calculate metrics
        mae = mean_absolute_error(y_true_original, y_pred_original)
        mse = mean_squared_error(y_true_original, y_pred_original)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_original, y_pred_original)
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS:")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")
        print("="*50 + "\n")
        
        # Save results to file
        model_name_short = model_filename.replace('.pkl', '')
        results_file = os.path.join(output_folder, f"eval_results_{model_name_short}.txt")
        with open(results_file, 'w') as f:
            f.write(f"Model: {model_filename}\n")
            f.write(f"Architecture: {model_architecture}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Test set size: {X_test.shape[0]}\n")
            f.write(f"Predictions count: {len(preds)}\n\n")
            f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
            f.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
            f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
            f.write(f"R² Score: {r2:.4f}\n")
        
        # Create a comparison DataFrame and save to CSV
        comparison_df = pd.DataFrame({
            'Predictions': y_pred_original,
            'Ground_Truth': y_true_original
        })
        csv_path = os.path.join(output_folder, f"predictions_vs_groundtruth_{model_name_short}.csv")
        comparison_df.to_csv(csv_path, index=False)
        
        # Create only the time series plot
        create_timeseries_plot(y_true_original, y_pred_original, model_name_short, output_folder)
        
        print(f"Results and time series plot saved to {output_folder}")
        return mae, mse, rmse, r2
    
    except Exception as e:
        print(f"Error evaluating model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_timeseries_plot(y_true, y_pred, model_name, output_folder):
    """Create only the time series plot"""
    # Time Series Plot
    plt.figure(figsize=(12, 6))
    indices = np.arange(len(y_true))
    plt.plot(indices, y_true, label='Actual', alpha=0.7)
    plt.plot(indices, y_pred, label='Predicted', alpha=0.7)
    plt.title('Actual vs Predicted Values Over Time')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_folder, f"timeseries_{model_name}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Created time series plot: timeseries_{model_name}.png")

def main():
    parser = argparse.ArgumentParser(description='Evaluation script that only produces the time series plot')
    parser.add_argument('model_architecture', type=str, choices=['TSTPlus', 'TSiTPlus', 'ResNetPlus'],
                        help='Model architecture (TSTPlus, TSiTPlus, or ResNetPlus)')
    parser.add_argument('dataset_name', type=str, help='Dataset name (e.g., dataset1)')
    parser.add_argument('model_filename', type=str, help='Model filename (e.g., reg_TSTPlus_dataset1_trial56_bs64_dr0.025_wd0.159_lr0.008458_optLamb_mae4.4930.pkl)')
    
    args = parser.parse_args()
    
    evaluate_model(args.model_architecture, args.dataset_name, args.model_filename)

if __name__ == "__main__":
    main()

#ALL TSTPLUS
#6Hours
#python evaluate_tsai.py TSTPlus dataset1 reg_TSTPlus_dataset1_trial56_bs64_dr0.025_wd0.159_lr0.008458_optLamb_mae4.4930.pkl
#python evaluate_tsai.py TSTPlus dataset2 reg_TSTPlus_dataset2_bs64_dr0.132381_wd0.1098056_lr0.0268906_optLamb_mae_4.3311.pkl
#python evaluate_tsai.py TSTPlus dataset3 reg_TSTPlus_dataset3_bs64_dr0.103519_wd0.091335_lr0.031357_optLamb_mae_4.3039.pkl
#python evaluate_tsai.py TSTPlus dataset4 reg_TSTPlus_dataset4_trial32_bs32_dr0.240_wd0.020_lr0.008941_optAdam_mae4.3837.pkl
#python evaluate_tsai.py TSTPlus dataset5 reg_TSTPlus_dataset5_trial12_bs64_dr0.130_wd0.040_lr0.012171_optAdam_mae4.2703.pkl

#8Hours
#python evaluate_tsai.py TSTPlus dataset3 reg_TSTPlus_dataset3_trial48_bs64_dr0.161_wd0.123_lr0.037091_optLamb_mae5.0805.pkl

#10hours
#python evaluate_tsai.py TSTPlus dataset3 reg_TSTPlus_dataset3_trial24_bs32_dr0.061_wd0.163_lr0.049329_optLamb_mae5.2987.pkl

#12hours
#python evaluate_tsai.py TSTPlus dataset3 reg_TSTPlus_dataset3_trial15_bs64_dr0.238_wd0.130_lr0.049872_optLamb_mae5.3960.pkl



# RESNETPLUS
#python evaluate_tsai.py ResNetPlus dataset5 reg_ResNetPlus_dataset5_trial43_bs32_dr0.026_wd0.003_lr0.024961_optLamb_mae4.2999.pkl

# TSiTPlus
# python evaluate_tsai.py TSiTPlus dataset4 reg_TSiTPlus_dataset4_trial15_bs32_dr0.139_wd0.083_lr0.021309_optLamb_mae4.7489.pkl


### Waiting Count
#python evaluate_tsai.py TSiTPlus dataset15 reg_TSiTPlus_dataset15_bs32_dr0.2_wd0.15_lr0.005_optSGD_mae_4.1953.pkl