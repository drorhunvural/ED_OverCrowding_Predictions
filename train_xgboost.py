import os
import numpy as np
import joblib
import argparse
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from datetime import datetime
from itertools import product

# Consistent path variables like RF code
dataset_folder = "datasets"
log_subfolder = "traditional/logs"
trained_folder = "traditional/trainedfiles"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train XGBoost Model')
    parser.add_argument('-d', '--dataset', type=str, required=True, help="Name of the dataset folder (e.g., dataset1, dataset2)")
    return parser.parse_args()

def load_datasets(dataset_dir, dataset_name):
    X_train = np.load(os.path.join(dataset_dir, f'X_train_{dataset_name}.npy'))
    y_train = np.load(os.path.join(dataset_dir, f'y_train_{dataset_name}.npy'))
    X_val = np.load(os.path.join(dataset_dir, f'X_val_{dataset_name}.npy'))
    y_val = np.load(os.path.join(dataset_dir, f'y_val_{dataset_name}.npy'))
    X_test = np.load(os.path.join(dataset_dir, f'X_test_{dataset_name}.npy'))
    y_test = np.load(os.path.join(dataset_dir, f'y_test_{dataset_name}.npy'))
    target_scaler = joblib.load(os.path.join(dataset_dir, f'target_scaler_{dataset_name}.pkl'))
    # Concatenate training and validation data
    X_train = np.concatenate((X_train, X_val), axis=0)
    y_train = np.concatenate((y_train, y_val), axis=0)
    return X_train, y_train, X_test, y_test, target_scaler

def main():
    args = parse_arguments()
    # Set directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, dataset_folder, args.dataset)
    log_dir = os.path.join(current_dir, log_subfolder)
    trained_models_dir = os.path.join(current_dir, trained_folder)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(trained_models_dir, exist_ok=True)

    # Load data
    X_train, y_train, X_test, y_test, target_scaler = load_datasets(dataset_dir, args.dataset)
    X_train = X_train.reshape(X_train.shape[0], -1)  # Flatten to (samples, features)
    X_test = X_test.reshape(X_test.shape[0], -1)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)

    # Expanded hyperparameter grid
    hyperparameter_grid = {
        'n_estimators': [100, 300],  # Number of boosting rounds
        'max_depth': [10, 12],       # Tree depth
        'eta': [0.01, 0.02],         # Learning rate
        'subsample': [0.8, 1.0],     # Row sampling
        'colsample_bytree': [0.3, 0.7],  # Column sampling
        'gamma': [0, 0.1],           # Min loss reduction
        'min_child_weight': [1, 3],  # Min hessian
        'lambda': [1, 1.5],          # L2 reg
        'alpha': [0.5, 1],           # L1 reg
        'scale_pos_weight': [1],     # For imbalance
        'tree_method': ['approx', 'hist'],  # Try 'hist' for speed
        'grow_policy': ['depthwise', 'lossguide'],
        'booster': ['dart', 'gbtree'],
        'normalize_type': ['forest', 'tree'],   # Only for dart
        'sample_type': ['uniform', 'weighted'], # Only for dart
        'rate_drop': [0.0, 0.3],    # Only for dart
        'one_drop': [0, 1],         # Only for dart
        'objective': ['reg:squarederror'],
        'eval_metric': ['rmse', 'mae'],
        'seed': [42],
        'max_delta_step': [0],      # Can try [0, 1] if highly imbalanced
        'verbosity': [0]
    }

    log_file_path = os.path.join(log_dir, 'results_log_XGBoost.txt')

    # Get all parameter combinations
    keys = list(hyperparameter_grid.keys())
    for values in product(*[hyperparameter_grid[k] for k in keys]):
        hp = dict(zip(keys, values))

        params = {
            'objective': hp['objective'],
            'eval_metric': hp['eval_metric'],
            'max_depth': hp['max_depth'],
            'eta': hp['eta'],
            'subsample': hp['subsample'],
            'colsample_bytree': hp['colsample_bytree'],
            'gamma': hp['gamma'],
            'min_child_weight': hp['min_child_weight'],
            'lambda': hp['lambda'],
            'alpha': hp['alpha'],
            'scale_pos_weight': hp['scale_pos_weight'],
            'tree_method': hp['tree_method'],
            'grow_policy': hp['grow_policy'],
            'booster': hp['booster'],
            'seed': hp['seed'],
            'max_delta_step': hp['max_delta_step'],
            'verbosity': hp['verbosity'],
        }
        # Add dart-specific params only if booster is dart
        if hp['booster'] == 'dart':
            params['normalize_type'] = hp['normalize_type']
            params['sample_type'] = hp['sample_type']
            params['rate_drop'] = hp['rate_drop']
            params['one_drop'] = hp['one_drop']

        num_boost_round = hp['n_estimators']
        evals = [(dtrain, 'train')]
        model = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=evals, verbose_eval=False)

        # Predict and inverse transform predictions to original scale
        y_pred_scaled = model.predict(dtest)
        y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_original = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))

        mse = mean_squared_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_original, y_pred_original)
        r2_test = r2_score(y_test_original, y_pred_original)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mae_str = f"{mae:.4f}".replace('.', '_')
        hp_str = "_".join([f"{k}_{str(v)}" for k, v in hp.items()])
        model_id_base = f"{timestamp}_{mae_str}_XGB_{hp_str}_{args.dataset}"

        # File names
        preds_csv_name = f"{model_id_base}.csv"
        preds_csv_path = os.path.join(log_dir, preds_csv_name)
        model_save_name = f"{model_id_base}.pkl"
        model_save_path = os.path.join(trained_models_dir, model_save_name)

        # 1. Save predictions
        df_preds = pd.DataFrame({
            "y_true": y_test_original.flatten(),
            "y_pred": y_pred_original.flatten()
        })
        df_preds.to_csv(preds_csv_path, index=False)
        # 2. Save model
        joblib.dump(model, model_save_path)
        # 3. Log
        with open(log_file_path, 'a') as log_file:
            log_file.write(
                f"Model = XGB, id={model_id_base}, dataset = {args.dataset}, " +
                ", ".join([f"{k}={v}" for k, v in hp.items()]) +
                f", MAE= {mae:.4f}, MSE= {mse:.4f}, RMSE= {rmse:.4f}, R2= {r2_test:.4f}\n"
            )
        print(f"MAE= {mae:.4f}, MSE= {mse:.4f}, RMSE= {rmse:.4f}, R2= {r2_test:.4f}\n")
        print(f"Saved prediction CSV: {preds_csv_name}")
        print(f"Saved model: {model_save_name}")

    print("Training complete.")

if __name__ == "__main__":
    main()
