import numpy as np
import torch
from tsai.basics import *
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import optuna
from optuna.pruners import PatientPruner, MedianPruner
from optuna.exceptions import TrialPruned
import joblib
import time
from fastai.callback.core import Callback
import random
import sys
import warnings
import pandas as pd
import argparse
warnings.filterwarnings('ignore')

# Define paths and folders for datasets, logs, and trained models
BASE_PATH = "tsai" 
LOG_FOLDER = os.path.join(BASE_PATH, "logs")
DATASETS_FOLDER = os.path.join(BASE_PATH, "datasets")
TRAINED_FOLDER = os.path.join(BASE_PATH, "trainedfiles")

# Model names to evaluate
MODEL_NAMES = ['TSTPlus']
EPOCHS = 20
MAX_LOSS_THRESHOLD = 100  # Threshold for pruning based on extreme MSE values

# Set a random seed for reproducibility
def set_complete_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Set seeds for fastai if applicable
    try:
        from fastai.torch_core import set_seed
        set_seed(seed)
    except ImportError:
        pass

# Initialize seed
set_complete_random_seed(42)

# Function to get available datasets in the hierarchical structure
def get_datasets(base_path):
    """
    Scan for datasets in the structure: base_path/Xhours/datasetY/
    Returns: {dataset_path: dataset_name} mapping
    """
    datasets = {}
    
    if not os.path.exists(base_path):
        print(f"Warning: Base path {base_path} does not exist!")
        return datasets
    
    # Scan for hour folders (e.g., 6hours, 8hours)
    for hour_entry in os.scandir(base_path):
        if hour_entry.is_dir() and hour_entry.name.endswith('hours'):
            hour_folder = hour_entry.path
            
            # Scan for dataset folders within each hour folder
            for dataset_entry in os.scandir(hour_folder):
                if dataset_entry.is_dir() and not dataset_entry.name.startswith("."):
                    dataset_path = dataset_entry.path
                    # Create a unique name including the hour info
                    dataset_name = f"{hour_entry.name}_{dataset_entry.name}"
                    datasets[dataset_path] = dataset_name
    
    return datasets

# Function to load datasets using .npy files
def load_dataset(dataset_path, data_type):
    """Load dataset from .npy files"""
    dataset_folder_name = os.path.basename(dataset_path)
    file_path = os.path.join(dataset_path, f"{data_type}_{dataset_folder_name}.npy")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    print(f"Loading dataset from {file_path}")
    return np.load(file_path)

# Enhanced logging function with detailed parameter and metric tracking
def log_trial_results(model_name, dataset_name, trial_num, params, metrics=None, status="pruned", 
                      reason="", runtime=0.0):
    """Log the results of a trial with detailed formatting"""
    # Create dataset-specific log folder
    dataset_log_folder = os.path.join(LOG_FOLDER, dataset_name)
    os.makedirs(dataset_log_folder, exist_ok=True)
    
    log_file_path = os.path.join(dataset_log_folder, f'results_log_{model_name}.txt')
    
    # Get parameter values with proper formatting
    lr_val = params.get('lr', 'N/A')
    lr_str = f"{lr_val:.6f}" if isinstance(lr_val, float) else "N/A"
    
    dropout_val = params.get('dropout', 'N/A')
    dropout_str = f"{dropout_val:.3f}" if isinstance(dropout_val, float) else "N/A"
    
    wd_val = params.get('wd', 'N/A')
    wd_str = f"{wd_val:.3f}" if isinstance(wd_val, float) else "N/A"
    
    opt_func_val = params.get('opt_func', 'N/A')
    opt_func_str = opt_func_val.__name__ if hasattr(opt_func_val, '__name__') else "N/A"
    
    # Construct the detailed log message
    if metrics is not None:  # Completed trial with metrics
        mae_val = metrics.get('mae', 'N/A')
        mae_str = f"{mae_val:.4f}" if isinstance(mae_val, float) else "N/A"
        
        mse_val = metrics.get('mse', 'N/A')
        mse_str = f"{mse_val:.4f}" if isinstance(mse_val, float) else "N/A"
        
        rmse_val = metrics.get('rmse', 'N/A')
        rmse_str = f"{rmse_val:.4f}" if isinstance(rmse_val, float) else "N/A"
        
        r2_val = metrics.get('r2', 'N/A')
        r2_str = f"{r2_val:.4f}" if isinstance(r2_val, float) else "N/A"
        
        log_message = (
            f"Trial={trial_num}, Status={status}, Model={model_name}, Dataset={dataset_name}, "
            f"Epochs={EPOCHS}, Batch_size={params.get('batchsize', 'N/A')}, "
            f"LR={lr_str}, Dropout={dropout_str}, Weight_decay={wd_str}, Opt_func={opt_func_str}, "
            f"Fusion_act={params.get('fusion_act', 'N/A')}, Fusion_layers={params.get('fusion_layers', 'N/A')}, "
            f"MAE={mae_str}, MSE={mse_str}, RMSE={rmse_str}, R2={r2_str}, "
            f"Runtime={runtime:.2f} minutes\n"
        )
    else:  # Pruned trial or error
        log_message = (
            f"Trial={trial_num}, Status={status}, Model={model_name}, Dataset={dataset_name}, "
            f"Epochs={EPOCHS}, Batch_size={params.get('batchsize', 'N/A')}, "
            f"LR={lr_str}, Dropout={dropout_str}, Weight_decay={wd_str}, Opt_func={opt_func_str}, "
            f"Fusion_act={params.get('fusion_act', 'N/A')}, Fusion_layers={params.get('fusion_layers', 'N/A')}, "
            f"Reason=\"{reason}\", Runtime={runtime:.2f} minutes\n"
        )
    
    # Write the log
    with open(log_file_path, "a") as log_file:
        log_file.write(log_message)
    
    print(f"Logged trial results to {log_file_path}")
    return log_message

# Custom MSE function with better error handling and proper CPU conversion
def mean_squared_error_fastai(y_pred, y_true):
    try:
        # Convert to CPU first, then to numpy
        y_pred, y_true = y_pred.cpu().detach().numpy(), y_true.cpu().detach().numpy()
        
        # Check for NaN or infinite values
        if np.isnan(y_pred).any() or np.isnan(y_true).any():
            print("Warning: NaN values detected in predictions or targets")
            raise TrialPruned("NaN encountered in predictions or targets.")
            
        if np.isinf(y_pred).any() or np.isinf(y_true).any():
            print("Warning: Infinite values detected in predictions or targets")
            raise TrialPruned("Infinite values encountered in predictions or targets.")
        
        # Calculate MSE
        mse = mean_squared_error(y_true, y_pred)
        
        # Check if MSE is too high
        if mse > MAX_LOSS_THRESHOLD:
            print(f"Warning: MSE value too high ({mse})")
            raise TrialPruned(f"MSE value too high: {mse}")
        
        return mse
    except Exception as e:
        print(f"Error in MSE calculation: {str(e)}")
        raise TrialPruned(f"Error in MSE calculation: {str(e)}")

# Custom callback to monitor training metrics and prune bad trials early
class TrainingMonitorCallback(Callback):
    def __init__(self, trial, threshold=MAX_LOSS_THRESHOLD):
        super().__init__()
        self.trial = trial
        self.threshold = threshold
        self.best_loss = float('inf')
        self.stagnation_counter = 0
        self.max_stagnation = 10  # Number of epochs with no improvement before pruning
    
    def after_batch(self):
        # Make sure to move tensor to CPU before checking values
        current_loss = self.loss
        if isinstance(current_loss, torch.Tensor):
            current_loss = current_loss.cpu().item()  # Safe conversion to Python scalar
            
        # Check for NaN or Inf in the loss after each batch
        if np.isnan(current_loss) or np.isinf(current_loss):
            print(f"Training loss is {current_loss}, pruning trial")
            raise TrialPruned("Training loss is NaN or Inf")
            
        # Check if loss is too high
        if current_loss > self.threshold:
            print(f"Training loss too high ({current_loss}), pruning trial")
            raise TrialPruned(f"Training loss too high: {current_loss}")
    
    def after_epoch(self):
        # Get current training loss, ensuring it's a Python scalar
        if len(self.learn.recorder.losses) > 0:
            last_loss = self.learn.recorder.losses[-1]
            if isinstance(last_loss, torch.Tensor):
                train_loss = last_loss.cpu().item()
            else:
                train_loss = float(last_loss)
        else:
            train_loss = float('inf')
        
        # Report to Optuna
        self.trial.report(train_loss, self.epoch)
        
        # Check for improvement
        if train_loss < self.best_loss * 0.999:  # 0.1% improvement
            self.best_loss = train_loss
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
            
        # Check for stagnation
        if self.stagnation_counter >= self.max_stagnation:
            print(f"Training loss stagnated for {self.max_stagnation} epochs, pruning trial")
            raise TrialPruned("Training loss stagnated")
        
        # Let Optuna decide if we should prune
        if self.trial.should_prune():
            print("Trial pruned by Optuna")
            raise TrialPruned()

# Optuna objective function
def create_objective(model_name, dataset_path, dataset_name):
    def objective(trial):
        iteration_start_time = time.time()
        
        # Initialize parameters dict - will be used even if trial is pruned
        current_params = {}
        reg = None  # Initialize reg variable at function scope
        
        try:
            # Hyperparameter suggestions
            lr = trial.suggest_float("lr", 1e-4, 0.05, log=True)
            dropout = trial.suggest_float("dropout", 0.0, 0.3)
            wd = trial.suggest_float("wd", 0.0, 0.2)
            opt_func = trial.suggest_categorical("opt_func", [Adam, Lamb])
            fusion_act = trial.suggest_categorical("fusion_act", ['relu', 'gelu', 'silu'])
            batchsize = trial.suggest_categorical("batchsize", [32, 64])
            fusion_layers = trial.suggest_int("fusion_layers", 128, 512, step=64)
            
            # Store parameters for logging
            current_params = {
                "lr": lr,
                "dropout": dropout,
                "wd": wd,
                "opt_func": opt_func,
                "fusion_act": fusion_act,
                "batchsize": batchsize,
                "fusion_layers": fusion_layers
            }
            
            print(f"Trial {trial.number}: Starting with parameters: lr={lr}, dropout={dropout}, "
                  f"wd={wd}, opt={opt_func.__name__}, act={fusion_act}, bs={batchsize}, "
                  f"layers={fusion_layers}")
            
            # Load datasets from the specified path
            X_train = load_dataset(dataset_path, 'train_X')
            X_val = load_dataset(dataset_path, 'val_X')
            y_train = load_dataset(dataset_path, 'train_y')
            y_val = load_dataset(dataset_path, 'val_y')
            X_test = load_dataset(dataset_path, 'test_X')
            y_test = load_dataset(dataset_path, 'test_y')
            
            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
            print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
            
            # Prepare data splits
            X, y, splits = combine_split_data([X_train, X_val], [y_train, y_val])
            
            # Model definition
            reg = TSRegressor(
                X, y,
                splits=splits,
                arch=model_name,
                metrics=mean_squared_error_fastai,
                bs=batchsize,
                lr=lr,
                opt_func=opt_func,
                fusion_dropout=dropout,
                wd=wd,
                fusion_act=fusion_act,
                fusion_layers=fusion_layers,
                seed=42 
            )
            
            # Ensure log directories exist
            dataset_log_folder = os.path.join(LOG_FOLDER, dataset_name)
            os.makedirs(dataset_log_folder, exist_ok=True)
            
            # Create and add the monitoring callback
            monitor_cb = TrainingMonitorCallback(trial)
            
            print(f"Starting training for {EPOCHS} epochs")
            
            try:
                # Training with monitoring - catch pruning exceptions here
                reg.fit_one_cycle(EPOCHS, lr, cbs=[monitor_cb])
                print("Training completed successfully")
            except TrialPruned as e:
                print(f"Training was pruned: {str(e)}")
                # Continue with evaluation despite pruning
                
            # Evaluate model on test set regardless of whether pruned or not
            print("Evaluating on test set")
            
            # Predict on test data
            raw_preds, target, preds = reg.get_X_preds(X_test, y_test)
            
            # Make sure predictions are on CPU and converted to numpy properly
            if isinstance(preds, torch.Tensor):
                preds = preds.cpu().detach().numpy()
            else:
                preds = np.asarray(preds)
            
            # Load target scaler from the correct path
            dataset_folder_name = os.path.basename(dataset_path)
            scaler_path = os.path.join(dataset_path, f'target_scaler_{dataset_folder_name}.pkl')
            
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Target scaler not found: {scaler_path}")
            
            target_scaler = joblib.load(scaler_path)
            y_true_original = target_scaler.inverse_transform(y_test.reshape(-1, 1))
            y_pred_original = target_scaler.inverse_transform(preds.reshape(-1, 1))
            
            # Calculate metrics
            mae = mean_absolute_error(y_true_original, y_pred_original)
            mse = mean_squared_error(y_true_original, y_pred_original)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true_original, y_pred_original)
            
            # Store metrics for logging
            metrics = {
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "r2": r2
            }
            
            print(f"Evaluation metrics - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
            
            # Save model with organized folder structure
            model_dir = os.path.join(TRAINED_FOLDER, model_name, dataset_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Create consistent filename format
            model_file = f"reg_{model_name}_{dataset_name}_trial{trial.number}_mae{mae:.4f}.pkl"
            model_path = os.path.join(model_dir, model_file)
            
            print(f"Saving model to {model_path}")
            reg.export(model_path)
            
            # Save CSV with ground truth and predictions
            csv_file = f"predictions_{model_name}_{dataset_name}_trial{trial.number}_mae{mae:.4f}.csv"
            csv_path = os.path.join(model_dir, csv_file)
            
            # Create DataFrame with ground truth and predictions
            predictions_df = pd.DataFrame({
                'y_test_ground_truth': y_true_original.flatten(),
                'y_pred': y_pred_original.flatten()
            })
            
            # Save to CSV
            predictions_df.to_csv(csv_path, index=False)
            print(f"Saved predictions: {csv_path}")
            
            # Verify that the model was saved
            if os.path.exists(model_path):
                print(f"Model saved successfully to {model_path}")
            else:
                print(f"WARNING: Model file not found at {model_path}")
            
            # Calculate runtime
            iteration_end_time = time.time()
            iteration_duration = (iteration_end_time - iteration_start_time) / 60.0  # in minutes
            
            # Determine trial status
            trial_status = "completed"
            reason = ""
            if 'TrialPruned' in str(sys.exc_info()[1]):
                trial_status = "pruned"
                reason = str(sys.exc_info()[1])
            
            # Log the trial with metrics in all cases
            log_trial_results(
                model_name=model_name,
                dataset_name=dataset_name,
                trial_num=trial.number,
                params=current_params,
                metrics=metrics,  # Always include metrics
                status=trial_status,
                reason=reason,
                runtime=iteration_duration
            )
            
            # If the trial was pruned, propagate the pruning exception
            if 'TrialPruned' in str(sys.exc_info()[1]):
                print(f"Trial {trial.number} pruned but evaluation metrics and model saved")
                raise TrialPruned(reason)
            
            print(f"Trial {trial.number} completed successfully")
            return mae
            
        except Exception as e:
            # Calculate runtime for failed trial
            iteration_end_time = time.time()
            iteration_duration = (iteration_end_time - iteration_start_time) / 60.0  # in minutes
            
            print(f"Trial {trial.number} failed with error: {str(e)}")
            
            # Log without metrics for failed trials
            log_trial_results(
                model_name=model_name,
                dataset_name=dataset_name,
                trial_num=trial.number,
                params=current_params,
                status="error",
                reason=str(e),
                runtime=iteration_duration
            )
            
            # Re-raise as TrialPruned to allow Optuna to continue
            raise TrialPruned(f"Trial pruned due to error: {str(e)}")
            
        finally:
            # Log execution time separately
            iteration_end_time = time.time()
            iteration_duration = (iteration_end_time - iteration_start_time) / 60.0  # in minutes
            dataset_log_folder = os.path.join(LOG_FOLDER, dataset_name)
            exec_time_log_path = os.path.join(dataset_log_folder, f'execution_time_{model_name}.txt')
            os.makedirs(dataset_log_folder, exist_ok=True)
            with open(exec_time_log_path, "a") as time_log:
                time_log.write(f"Trial {trial.number}: {iteration_duration:.2f} minutes - {'pruned' if isinstance(sys.exc_info()[1], TrialPruned) else 'completed'}\n")
    
    return objective

# Main execution function 
def main():
    parser = argparse.ArgumentParser(description='Train time series models with Optuna optimization')
    parser.add_argument('--hours', type=str, default='6hours',
                       help='Hour folder to use (e.g., 6hours, 8hours). Default: 6hours')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Specific dataset to train on (e.g., dataset1). If not provided, trains on all datasets in the hour folder.')
    args = parser.parse_args()
    
    print("Starting optimization process")
    print(f"Models to evaluate: {MODEL_NAMES}")
    print(f"Maximum epochs per trial: {EPOCHS}")
    print(f"MSE threshold for pruning: {MAX_LOSS_THRESHOLD}")
    print(f"Hour folder: {args.hours}")
    
    # Construct the base path for the specified hour folder
    hour_folder_path = os.path.join(DATASETS_FOLDER, args.hours)
    
    if not os.path.exists(hour_folder_path):
        print(f"Error: Hour folder '{hour_folder_path}' does not exist!")
        
        # Show available hour folders
        if os.path.exists(DATASETS_FOLDER):
            available_hours = [d for d in os.listdir(DATASETS_FOLDER) 
                             if os.path.isdir(os.path.join(DATASETS_FOLDER, d)) and d.endswith('hours')]
            if available_hours:
                print(f"Available hour folders in '{DATASETS_FOLDER}':")
                for hour_folder in sorted(available_hours):
                    print(f"  - {hour_folder}")
            else:
                print(f"No hour folders found in '{DATASETS_FOLDER}'")
        return
    
    # Determine which datasets to process
    if args.dataset:
        # Train on specific dataset
        specific_dataset_path = os.path.join(hour_folder_path, args.dataset)
        if not os.path.exists(specific_dataset_path):
            print(f"Error: Dataset folder '{specific_dataset_path}' does not exist!")
            
            # Show available datasets in the hour folder
            if os.path.exists(hour_folder_path):
                available_datasets = [d for d in os.listdir(hour_folder_path) 
                                    if os.path.isdir(os.path.join(hour_folder_path, d))]
                if available_datasets:
                    print(f"Available datasets in '{hour_folder_path}':")
                    for dataset in sorted(available_datasets):
                        print(f"  - {dataset}")
                else:
                    print(f"No datasets found in '{hour_folder_path}'")
            return
        
        dataset_paths = [(specific_dataset_path, f"{args.hours}_{args.dataset}")]
        print(f"Training on specific dataset: {args.dataset} in {args.hours}")
    else:
        # Train on all datasets in the hour folder
        print(f"Searching for all datasets in {hour_folder_path}...")
        dataset_paths = []
        
        if os.path.exists(hour_folder_path):
            for dataset_folder in os.listdir(hour_folder_path):
                dataset_path = os.path.join(hour_folder_path, dataset_folder)
                if os.path.isdir(dataset_path) and not dataset_folder.startswith('.'):
                    dataset_name = f"{args.hours}_{dataset_folder}"
                    dataset_paths.append((dataset_path, dataset_name))
        
        if not dataset_paths:
            print(f"No datasets found in '{hour_folder_path}'")
            return
        
        print(f"Found {len(dataset_paths)} datasets to process:")
        for _, name in dataset_paths:
            print(f"  - {name}")
    
    # Create necessary directories
    os.makedirs(LOG_FOLDER, exist_ok=True)
    os.makedirs(TRAINED_FOLDER, exist_ok=True)
    print(f"Created trained models directory at: {TRAINED_FOLDER}")
    
    for model_name in MODEL_NAMES:
        # Create model subdirectory for consistent structure
        model_subfolder = os.path.join(TRAINED_FOLDER, model_name)
        os.makedirs(model_subfolder, exist_ok=True)
        print(f"Created directory for {model_name} models at: {model_subfolder}")
    
    # Process each dataset
    for dataset_path, dataset_name in dataset_paths:
        print(f"\n{'='*60}")
        print(f"PROCESSING DATASET: {dataset_name}")
        print(f"Dataset path: {dataset_path}")
        print(f"{'='*60}")
        
        # Loop through each model for this dataset
        for model_name in MODEL_NAMES:
            print(f"\n--- Training {model_name} on {dataset_name} ---")
            
            # Create a robust pruner
            pruner = PatientPruner(MedianPruner(n_startup_trials=3, n_warmup_steps=5), patience=3)
            
            # Create study
            study = optuna.create_study(direction="minimize", pruner=pruner, 
                                       study_name=f"{model_name}_{dataset_name}_optimization")
            
            # Run optimization
            print(f"Starting optimization with 50 trials")
            study.optimize(create_objective(model_name, dataset_path, dataset_name), n_trials=2)
            
            # Save study results
            dataset_log_folder = os.path.join(LOG_FOLDER, dataset_name)
            os.makedirs(dataset_log_folder, exist_ok=True)
            study_file = os.path.join(dataset_log_folder, f"optuna_study_{model_name}_{dataset_name}.pkl")
            joblib.dump(study, study_file)
            
            # Check if we have completed trials before accessing best_trial
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if completed_trials:
                # Print and log best results
                print(f"\n✅ Optimization for {model_name} on {dataset_name} completed")
                print(f"Best MAE: {study.best_trial.value:.4f}")
                print(f"Best parameters: {study.best_trial.params}")
                
                with open(os.path.join(dataset_log_folder, f"best_params_{model_name}_{dataset_name}.txt"), "w") as f:
                    f.write(f"Best MAE: {study.best_trial.value:.4f}\n")
                    for param, value in study.best_trial.params.items():
                        f.write(f"{param}: {value}\n")
            else:
                print(f"\n❌ No trials for {model_name} on {dataset_name} completed successfully.")
                with open(os.path.join(dataset_log_folder, f"best_params_{model_name}_{dataset_name}.txt"), "w") as f:
                    f.write("No trials completed successfully.\n")

            # Plot optimization history (optional)
            try:
                if completed_trials:
                    from optuna.visualization import plot_optimization_history
                    fig = plot_optimization_history(study)
                    fig.write_image(os.path.join(dataset_log_folder, f"optimization_history_{model_name}_{dataset_name}.png"))
            except:
                print("Could not generate optimization history plot")
    
    print(f"\n🎉 Training completed for all datasets!")

if __name__ == "__main__":
    main()

# Usage Examples:
# python train.py --hours 6hours --dataset dataset1        # Train only on dataset1 in 6hours folder
# python train.py --hours 6hours                           # Train on all datasets in 6hours folder  
# python train.py --dataset dataset1                       # Train only on dataset1 in default 6hours folder
# python train.py                                          # Train on all datasets in default 6hours folder
# python train.py --hours 8hours --dataset dataset2       # Train only on dataset2 in 8hours folder