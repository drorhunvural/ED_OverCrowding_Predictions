#!/usr/bin/env python3
"""
General validation script for any pruned TSAI model
Works with any model file and dataset structure
"""

import numpy as np
import pandas as pd
import joblib
import torch
from tsai.basics import *
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import re
import argparse
from optuna.exceptions import TrialPruned

def mean_squared_error_fastai(y_pred, y_true):
    """
    Custom MSE function - required for loading TSAI models that used this metric
    """
    try:
        y_pred, y_true = y_pred.cpu().detach().numpy(), y_true.cpu().detach().numpy()
        
        if np.isnan(y_pred).any() or np.isnan(y_true).any():
            print("Warning: NaN values detected in predictions or targets")
            raise TrialPruned("NaN encountered in predictions or targets.")
            
        if np.isinf(y_pred).any() or np.isinf(y_true).any():
            print("Warning: Infinite values detected in predictions or targets")
            raise TrialPruned("Infinite values encountered in predictions or targets.")
        
        mse = mean_squared_error(y_true, y_pred)
        
        MAX_LOSS_THRESHOLD = 100
        if mse > MAX_LOSS_THRESHOLD:
            print(f"Warning: MSE value too high ({mse})")
            raise TrialPruned(f"MSE value too high: {mse}")
        
        return mse
    except TrialPruned:
        raise
    except Exception as e:
        error_msg = f"Error in MSE calculation: {str(e)}"
        print(error_msg)
        raise TrialPruned(error_msg)

def extract_mae_from_filename(filename):
    """
    Extract MAE value from model filename
    Looks for patterns like 'mae4.2947' or 'MAE_4.2947'
    """
    # Look for mae followed by number (case insensitive)
    mae_pattern = r'mae[_]?(\d+\.?\d*)'
    match = re.search(mae_pattern, filename.lower())
    
    if match:
        return float(match.group(1))
    return None

def auto_detect_dataset_structure(dataset_path):
    """
    Automatically detect dataset file structure
    Returns dict with found files and dataset name
    """
    if not os.path.exists(dataset_path):
        return None
    
    files = os.listdir(dataset_path)
    
    # Try to find test files
    test_x_files = [f for f in files if f.startswith('test_X') and f.endswith('.npy')]
    test_y_files = [f for f in files if f.startswith('test_y') and f.endswith('.npy')]
    scaler_files = [f for f in files if 'scaler' in f.lower() and f.endswith('.pkl')]
    
    if not test_x_files or not test_y_files:
        return None
    
    # Extract dataset name from first file
    test_x_file = test_x_files[0]
    # Pattern: test_X_datasetname.npy -> extract datasetname
    dataset_name_match = re.search(r'test_X_(.+)\.npy', test_x_file)
    if dataset_name_match:
        dataset_name = dataset_name_match.group(1)
    else:
        dataset_name = "unknown"
    
    return {
        'dataset_name': dataset_name,
        'test_x_file': test_x_file,
        'test_y_file': test_y_files[0],
        'scaler_files': scaler_files
    }

def find_matching_scaler(scaler_files, dataset_name):
    """
    Find the appropriate scaler file for the dataset
    """
    # Look for target scaler first
    target_scalers = [f for f in scaler_files if 'target' in f.lower()]
    if target_scalers:
        return target_scalers[0]
    
    # Look for scaler with dataset name
    name_scalers = [f for f in scaler_files if dataset_name in f]
    if name_scalers:
        return name_scalers[0]
    
    # Return first scaler if available
    if scaler_files:
        return scaler_files[0]
    
    return None

def validate_model(model_path, dataset_path=None, verbose=True):
    """
    General model validation function
    
    Args:
        model_path: Path to the .pkl model file
        dataset_path: Path to dataset folder (auto-detected if None)
        verbose: Whether to print detailed output
    
    Returns:
        dict: Validation results
    """
    results = {
        'success': False,
        'mae': None,
        'mse': None,
        'rmse': None,
        'r2': None,
        'expected_mae': None,
        'mae_matches': False,
        'issues': [],
        'recommendation': 'UNKNOWN'
    }
    
    if verbose:
        print("🔍 VALIDATING MODEL")
        print("="*60)
        print(f"Model: {os.path.basename(model_path)}")
    
    # Extract expected MAE from filename
    expected_mae = extract_mae_from_filename(os.path.basename(model_path))
    results['expected_mae'] = expected_mae
    
    if verbose and expected_mae:
        print(f"Expected MAE from filename: {expected_mae:.4f}")
    
    # Auto-detect dataset path if not provided
    if dataset_path is None:
        if verbose:
            print("🔍 Auto-detecting dataset path...")
        
        # Extract dataset info from model filename
        model_filename = os.path.basename(model_path)
        
        # Parse the filename to extract components
        # Expected pattern: reg_ModelName_HoursFeature_DatasetName_trial_mae.pkl
        # Example: reg_TSTPlus_6hours_boarding_count_dataset1_trial4_mae4.2947.pkl
        
        dataset_info = {}
        
        # Try to extract hours_feature and dataset components
        # Pattern 1: Look for hours_feature_dataset pattern
        hours_dataset_pattern = r'(\d+hours_[^_]+)_(dataset\d+)'
        match = re.search(hours_dataset_pattern, model_filename)
        
        if match:
            dataset_info['hours_feature'] = match.group(1)  # e.g., "6hours_boarding_count"
            dataset_info['dataset_name'] = match.group(2)   # e.g., "dataset1"
            
            # Construct expected path: datasets/hours_feature/dataset_name/
            expected_path = os.path.join('datasets', dataset_info['hours_feature'], dataset_info['dataset_name'])
            
            if verbose:
                print(f"   Extracted: {dataset_info['hours_feature']} / {dataset_info['dataset_name']}")
                print(f"   Expected path: {expected_path}")
            
            # Check if this path exists
            if os.path.exists(expected_path):
                # Verify it has test files
                test_files = [f for f in os.listdir(expected_path) 
                            if f.startswith('test_') and f.endswith('.npy')]
                if test_files:
                    dataset_path = expected_path
                    if verbose:
                        print(f"   ✅ Found dataset: {dataset_path}")
        
        # If still not found, try alternative patterns and searches
        if not dataset_path:
            if verbose:
                print("   Primary pattern failed, trying alternative searches...")
            
            # Fallback: search for any matching components
            potential_components = []
            
            # Extract potential dataset identifiers
            patterns = [
                r'(\d+hours_[^_]+_dataset\d+)',  # Full pattern
                r'(\d+hours_[^_]+)',             # Just hours_feature
                r'(dataset\d+)',                  # Just dataset
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, model_filename)
                potential_components.extend(matches)
            
            # Search in datasets folder
            if os.path.exists('datasets'):
                for hours_folder in os.listdir('datasets'):
                    hours_path = os.path.join('datasets', hours_folder)
                    if os.path.isdir(hours_path) and 'hours_' in hours_folder:
                        if verbose:
                            print(f"   Checking: {hours_path}")
                        
                        # Check if any component matches this hours folder
                        for component in potential_components:
                            if component in hours_folder or hours_folder in component:
                                # Look for dataset folders within
                                for dataset_folder in os.listdir(hours_path):
                                    dataset_full_path = os.path.join(hours_path, dataset_folder)
                                    if os.path.isdir(dataset_full_path):
                                        # Check if this matches dataset pattern or any component
                                        for comp in potential_components:
                                            if comp in dataset_folder or dataset_folder in comp:
                                                # Verify test files exist
                                                test_files = [f for f in os.listdir(dataset_full_path) 
                                                            if f.startswith('test_') and f.endswith('.npy')]
                                                if test_files:
                                                    dataset_path = dataset_full_path
                                                    if verbose:
                                                        print(f"   ✅ Found dataset: {dataset_path}")
                                                    break
                                        if dataset_path:
                                            break
                                if dataset_path:
                                    break
                        if dataset_path:
                            break
    
    if not dataset_path:
        if verbose:
            print("❌ Dataset path could not be auto-detected")
            print("\n💡 Available options:")
            print("1. Specify dataset path manually:")
            print(f"   python validate_model.py {model_path} --dataset_path /path/to/dataset")
            print("\n2. Available directories to check:")
            
            # Show available directories that might contain datasets
            for base_dir in ['datasets', 'data', '.']:
                if os.path.exists(base_dir):
                    print(f"\n   In {base_dir}/:")
                    try:
                        for item in sorted(os.listdir(base_dir)):
                            item_path = os.path.join(base_dir, item)
                            if os.path.isdir(item_path):
                                if 'hours_' in item or 'dataset' in item.lower():
                                    print(f"     - {item}/")
                                    # Show subdirectories
                                    try:
                                        sub_items = [s for s in os.listdir(item_path) 
                                                   if os.path.isdir(os.path.join(item_path, s))][:5]
                                        if sub_items:
                                            print(f"       Contains: {', '.join(sub_items)}")
                                    except:
                                        pass
                    except PermissionError:
                        print(f"     Permission denied")
        return results
    
    if verbose:
        print(f"Dataset: {dataset_path}")
    
    # Check if files exist
    if not os.path.exists(model_path):
        if verbose:
            print(f"❌ Model file not found: {model_path}")
        return results
    
    # Auto-detect dataset structure
    dataset_info = auto_detect_dataset_structure(dataset_path)
    if not dataset_info:
        if verbose:
            print(f"❌ Could not detect dataset structure in: {dataset_path}")
            print("Available files:")
            if os.path.exists(dataset_path):
                for f in os.listdir(dataset_path):
                    print(f"   - {f}")
        return results
    
    if verbose:
        print(f"✅ Dataset structure detected:")
        print(f"   Dataset name: {dataset_info['dataset_name']}")
        print(f"   Test X file: {dataset_info['test_x_file']}")
        print(f"   Test y file: {dataset_info['test_y_file']}")
        print(f"   Scaler files: {dataset_info['scaler_files']}")
    
    # Load model
    if verbose:
        print(f"\n🔄 Loading model...")
    try:
        reg = load_learner(model_path)
        if verbose:
            print("✅ Model loaded successfully!")
    except Exception as e:
        if verbose:
            print(f"❌ Error loading model: {e}")
        return results
    
    # Load test data
    if verbose:
        print(f"\n📊 Loading test data...")
    try:
        X_test = np.load(os.path.join(dataset_path, dataset_info['test_x_file']))
        y_test = np.load(os.path.join(dataset_path, dataset_info['test_y_file']))
        
        # Find and load scaler
        scaler_file = find_matching_scaler(dataset_info['scaler_files'], dataset_info['dataset_name'])
        if scaler_file:
            target_scaler = joblib.load(os.path.join(dataset_path, scaler_file))
            if verbose:
                print(f"✅ Data loaded successfully!")
                print(f"   X_test shape: {X_test.shape}")
                print(f"   y_test shape: {y_test.shape}")
                print(f"   Using scaler: {scaler_file}")
        else:
            if verbose:
                print(f"⚠️  No scaler file found - metrics will be on scaled data")
            target_scaler = None
        
    except Exception as e:
        if verbose:
            print(f"❌ Error loading test data: {e}")
        return results
    
    # Test predictions
    if verbose:
        print(f"\n🎯 Testing model predictions...")
    try:
        predictions_list = []
        
        for i in range(5):
            raw_preds, target, preds = reg.get_X_preds(X_test, y_test)
            if isinstance(preds, torch.Tensor):
                preds = preds.cpu().detach().numpy()
            predictions_list.append(preds.copy())
        
        # Check consistency
        pred_array = np.array(predictions_list)
        pred_std = np.std(pred_array, axis=0)
        max_std = np.max(pred_std)
        
        if verbose:
            if max_std > 0.001:
                print(f"⚠️  Predictions show inconsistency (max std: {max_std:.6f})")
            else:
                print(f"✅ Predictions are consistent (max std: {max_std:.6f})")
        
        preds = predictions_list[0]
        
    except Exception as e:
        if verbose:
            print(f"❌ Error during prediction: {e}")
        return results
    
    # Calculate metrics
    if verbose:
        print(f"\n📈 Calculating performance metrics...")
    try:
        if target_scaler:
            y_true_original = target_scaler.inverse_transform(y_test.reshape(-1, 1))
            y_pred_original = target_scaler.inverse_transform(preds.reshape(-1, 1))
        else:
            y_true_original = y_test.reshape(-1, 1)
            y_pred_original = preds.reshape(-1, 1)
        
        mae = mean_absolute_error(y_true_original, y_pred_original)
        mse = mean_squared_error(y_true_original, y_pred_original)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_original, y_pred_original)
        
        results.update({
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        })
        
        if verbose:
            print(f"   MAE:  {mae:.4f}")
            print(f"   MSE:  {mse:.4f}")
            print(f"   RMSE: {rmse:.4f}")
            print(f"   R²:   {r2:.4f}")
        
        # Compare with expected MAE
        if expected_mae:
            mae_diff = abs(mae - expected_mae)
            mae_matches = mae_diff < 0.01
            results['mae_matches'] = mae_matches
            
            if verbose:
                print(f"\n🔍 Comparison with filename:")
                print(f"   Expected MAE: {expected_mae:.4f}")
                print(f"   Actual MAE:   {mae:.4f}")
                print(f"   Difference:   {mae_diff:.4f}")
                
                if mae_matches:
                    print("✅ Results match filename!")
                else:
                    print("⚠️  Results differ from filename")
        
    except Exception as e:
        if verbose:
            print(f"❌ Error calculating metrics: {e}")
        return results
    
    # Check for issues
    if verbose:
        print(f"\n🔎 Checking for issues...")
    
    issues = []
    
    # Check for problematic values
    if np.isnan(y_pred_original).any():
        issues.append("NaN values in predictions")
    if np.isinf(y_pred_original).any():
        issues.append("Infinite values in predictions")
    if np.any(y_pred_original < 0):
        negative_count = np.sum(y_pred_original < 0)
        issues.append(f"Negative predictions ({negative_count}/{len(y_pred_original)})")
    
    # Check prediction vs true range
    pred_range = np.max(y_pred_original) - np.min(y_pred_original)
    true_range = np.max(y_true_original) - np.min(y_true_original)
    
    if verbose:
        print(f"   Prediction range: {np.min(y_pred_original):.2f} to {np.max(y_pred_original):.2f}")
        print(f"   True value range: {np.min(y_true_original):.2f} to {np.max(y_true_original):.2f}")
    
    if pred_range > 2 * true_range:
        issues.append("Prediction range much larger than true range")
    elif pred_range < 0.5 * true_range:
        issues.append("Prediction range much smaller than true range")
    
    # Residual analysis
    residuals = y_true_original.flatten() - y_pred_original.flatten()
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    
    if verbose:
        print(f"\n📊 Residual analysis:")
        print(f"   Mean residual: {residual_mean:.4f}")
        print(f"   Std residual:  {residual_std:.4f}")
    
    if abs(residual_mean) > 0.1 * residual_std:
        issues.append(f"Systematic bias detected (mean residual: {residual_mean:.4f})")
    
    results['issues'] = issues
    
    # Final recommendation
    if verbose:
        print(f"\n" + "="*60)
        print(f"🎯 FINAL ASSESSMENT")
        print(f"="*60)
    
    if issues:
        if verbose:
            print(f"⚠️  Issues found:")
            for issue in issues:
                print(f"   - {issue}")
        recommendation = "USE WITH CAUTION"
    else:
        if verbose:
            print(f"✅ No major issues detected")
        recommendation = "SAFE FOR PRODUCTION"
    
    results['recommendation'] = recommendation
    results['success'] = True
    
    if verbose:
        print(f"\n🏆 RECOMMENDATION: {recommendation}")
        
        if recommendation == "SAFE FOR PRODUCTION":
            print(f"""
✅ Model appears to be production-ready!
   - Stable and consistent predictions
   - Performance metrics are reasonable
   - No major issues detected
            """)
        else:
            print(f"""
⚠️  Review the issues above before production use.
            """)
    
    return results

def main():
    """
    Command line interface for model validation
    """
    parser = argparse.ArgumentParser(description='Validate any TSAI model (especially pruned ones)')
    parser.add_argument('model_path', help='Path to the .pkl model file')
    parser.add_argument('--dataset_path', help='Path to dataset folder (auto-detected if not provided)')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    results = validate_model(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        verbose=not args.quiet
    )
    
    if not results['success']:
        exit(1)
    
    if results['recommendation'] != "SAFE FOR PRODUCTION":
        exit(2)

if __name__ == "__main__":
    main()

#python validate_model.py tsai/trainedfiles/{trainedfile_name}.pkl