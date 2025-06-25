import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
from datetime import datetime
import joblib
from itertools import product
import sys

# If you want to guarantee your "models" folder is importable
this_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(this_dir, "models")
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

# ==== CLASSIC STATIC IMPORTS ====
from VanillaLSTMfunctional import VanillaLSTMfunctional
from VanillaLSTMsequential import VanillaLSTMsequential
from Seq2Seq import Seq2SeqLSTM
from BiLSTM import BiLSTM
# Add more models as you implement them

# ==== MODEL MAPPING ====
MODEL_CLASS_MAPPING = {
    'VanillaLSTMfunctional': VanillaLSTMfunctional,
    'VanillaLSTMsequential': VanillaLSTMsequential,
    'Seq2SeqLSTM': Seq2SeqLSTM,
    'BiLSTM': BiLSTM,
    # Add more if you add more models
}

# ==== FOLDER CONFIG ====
dataset_folder = os.path.join("dataset", "Dailysingletry")
log_subfolder = os.path.join("logs_patient_count", "Daily", "Daily5")
trained_folder = os.path.join("trainedfiles", "Daily")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train RNN-based model with hyperparameter grid')
    parser.add_argument('-m', '--model', type=str, required=True, help="Model to train (e.g., BiLSTM)")
    parser.add_argument('-d', '--dataset', type=str, required=True, help="Dataset name (e.g., dataset2)")
    return parser.parse_args()

def load_datasets(dataset_dir, dataset_name):
    X_train = np.load(os.path.join(dataset_dir, f'X_train_{dataset_name}.npy'))
    y_train = np.load(os.path.join(dataset_dir, f'y_train_{dataset_name}.npy'))
    X_val = np.load(os.path.join(dataset_dir, f'X_val_{dataset_name}.npy'))
    y_val = np.load(os.path.join(dataset_dir, f'y_val_{dataset_name}.npy'))
    X_test = np.load(os.path.join(dataset_dir, f'X_test_{dataset_name}.npy'))
    y_test = np.load(os.path.join(dataset_dir, f'y_test_{dataset_name}.npy'))
    target_scaler = joblib.load(os.path.join(dataset_dir, f'target_scaler_{dataset_name}.pkl'))
    return X_train, y_train, X_val, y_val, X_test, y_test, target_scaler

def run_grid_search(model_class, model_name, dataset_name, log_dir, trained_models_dir, param_grid):
    dataset_dir = os.path.join(this_dir, dataset_folder, dataset_name)
    log_file_path = os.path.join(log_dir, f'results_log_{model_name}.txt')
    X_train, y_train, X_val, y_val, X_test, y_test, target_scaler = load_datasets(dataset_dir, dataset_name)

    param_names = list(param_grid.keys())
    param_combinations = list(product(*param_grid.values()))

    print(f"\nTotal runs: {len(param_combinations)}\n")
    run_count = 0
    for param_values in param_combinations:
        params = dict(zip(param_names, param_values))
        run_count += 1
        print(f"\nRun {run_count}/{len(param_combinations)}: {params}")

        model = model_class()
        model.train(
            X_train, y_train, X_val, y_val,
            params['epochs'], params['batch_size'], params['patience'],
            params['dropout'], params['lstm_units'], params['optimizer'],
            params['loss'], params['lr'], params['weight_decay']
        )

        y_pred_scaled = model.predict(X_test)
        if len(y_pred_scaled.shape) == 3:
            y_pred_scaled = np.mean(y_pred_scaled, axis=1)
        y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_original = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))

        mse = mean_squared_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_original, y_pred_original)
        r2_test = r2_score(y_test_original, y_pred_original)

        print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2_test:.4f}")

        # Unique model/run ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        param_str = "_".join([f"{k}_{v}" for k, v in params.items()])
        model_id_base = f"{timestamp}_{param_str}_{model_name}_{dataset_name}"

        # Save predictions
        csv_filename = f"{model_id_base}_predictions.csv"
        csv_path = os.path.join(log_dir, csv_filename)
        prediction_df = pd.DataFrame({'y_true': y_test_original.flatten(), 'y_pred': y_pred_original.flatten()})
        prediction_df.to_csv(csv_path, index=False)

        # Save log
        with open(log_file_path, 'a') as log_file:
            log_file.write(
                f"Model={model_name}, dataset={dataset_name}, {param_str}, "
                f"MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2_test:.4f}\n"
            )

        try:
            model.save_model(model_dir=trained_models_dir, loss=mae, dataset_name=dataset_name)
        except Exception as e:
            print(f"Error saving model: {e}")

def main():
    args = parse_arguments()
    log_dir = os.path.join(this_dir, log_subfolder)
    trained_models_dir = os.path.join(this_dir, trained_folder)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(trained_models_dir, exist_ok=True)

    if args.model in MODEL_CLASS_MAPPING:
        model_class = MODEL_CLASS_MAPPING[args.model]
        # EXPANDED parameter grid for thorough search:
        param_grid = {
            'epochs': [100, 200],
            'batch_size': [16, 32, 64],
            'patience': [10, 30],
            'dropout': [0.1, 0.2, 0.3, 0.5],
            'lstm_units': [32, 50, 80, 128],
            'lr': [0.001, 0.005, 0.01],
            'weight_decay': [0.0, 0.01, 0.1],
            'optimizer': ["Adam", "SGD", "rmsprop"],
            'loss': ["mean_squared_error"]
        }
        run_grid_search(
            model_class=model_class,
            model_name=args.model,
            dataset_name=args.dataset,
            log_dir=log_dir,
            trained_models_dir=trained_models_dir,
            param_grid=param_grid
        )
    else:
        print(f"Model {args.model} is not implemented or mapped in MODEL_CLASS_MAPPING.")

if __name__ == "__main__":
    main()
