import os
import numpy as np
import joblib
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from itertools import product
from datetime import datetime

dataset_folder = "datasets"
log_subfolder = "traditional/logs"
trained_folder = "traditional/trainedfiles"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train RandomForest Model with extended grid search')
    parser.add_argument('-d', '--dataset', type=str, required=True, help="Name of the dataset folder (e.g., dataset1, dataset2)")
    return parser.parse_args()

def load_datasets(dataset_dir, dataset_name):
    try:
        print(f"Loading dataset from: {dataset_dir}")
        X_train = np.load(os.path.join(dataset_dir, f'X_train_{dataset_name}.npy'))
        y_train = np.load(os.path.join(dataset_dir, f'y_train_{dataset_name}.npy'))
        X_val = np.load(os.path.join(dataset_dir, f'X_val_{dataset_name}.npy'))
        y_val = np.load(os.path.join(dataset_dir, f'y_val_{dataset_name}.npy'))
        X_test = np.load(os.path.join(dataset_dir, f'X_test_{dataset_name}.npy'))
        y_test = np.load(os.path.join(dataset_dir, f'y_test_{dataset_name}.npy'))
        target_scaler = joblib.load(os.path.join(dataset_dir, f'target_scaler_{dataset_name}.pkl'))

        print(f"Dataset loaded successfully: {dataset_name}")

        # Concatenate training and validation data
        X_train = np.concatenate((X_train, X_val), axis=0)
        y_train = np.concatenate((y_train, y_val), axis=0)
    
        return X_train, y_train, X_test, y_test, target_scaler
    except Exception as e:
        print(f"Error loading datasets: {e}")
        exit(1)

def main():
    args = parse_arguments()
    print(f"Running training for dataset: {args.dataset}")
    
    # Set directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, dataset_folder, args.dataset)
    log_dir = os.path.join(current_dir, log_subfolder)
    trained_models_dir = os.path.join(current_dir, trained_folder)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(trained_models_dir, exist_ok=True)
    
    # Load data
    X_train, y_train, X_test, y_test, target_scaler = load_datasets(dataset_dir, args.dataset)
    X_train = X_train.reshape(X_train.shape[0], -1)  
    X_test = X_test.reshape(X_test.shape[0], -1)
    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    # Expanded parameter grid
    n_estimators_values = [200, 300, 400]
    max_depth_values = [20, 30, 40]
    min_samples_split_values = [2, 5, 10]
    min_samples_leaf_values = [2, 4, 8]
    bootstrap_values = [True, False]
    max_features_values = ['auto', 'sqrt', 0.5]  # You can also use float for percentage, e.g., 0.5 means 50%
    max_samples_values = [None, 0.7]  # Use None for default (all), or a float for percentage
    min_weight_fraction_leaf_values = [0.0, 0.01]
    oob_score_values = [False, True]
    ccp_alpha_values = [0.0, 0.01]
    criterion_values = ['squared_error', 'absolute_error']  # For regression

    log_file_path = os.path.join(log_dir, 'results_log_RandomForest.txt')
    with open(log_file_path, 'a') as log_file:
        for n_estimators, max_depth, min_samples_split, min_samples_leaf, bootstrap, max_features, max_samples, min_weight_fraction_leaf, oob_score, ccp_alpha, criterion in product(
            n_estimators_values,
            max_depth_values,
            min_samples_split_values,
            min_samples_leaf_values,
            bootstrap_values,
            max_features_values,
            max_samples_values,
            min_weight_fraction_leaf_values,
            oob_score_values,
            ccp_alpha_values,
            criterion_values
        ):
            print(f"Training model with n_estimators={n_estimators}, max_depth={max_depth}, "
                  f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, "
                  f"bootstrap={bootstrap}, max_features={max_features}, max_samples={max_samples}, "
                  f"min_weight_fraction_leaf={min_weight_fraction_leaf}, oob_score={oob_score}, "
                  f"ccp_alpha={ccp_alpha}, criterion={criterion}")

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                bootstrap=bootstrap,
                max_features=max_features,
                max_samples=max_samples,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                oob_score=oob_score,
                ccp_alpha=ccp_alpha,
                criterion=criterion,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train.ravel())

            y_pred_scaled = model.predict(X_test)
            y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1))
            y_pred_original = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))

            mse = mean_squared_error(y_test_original, y_pred_original)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_original, y_pred_original)
            r2_test = r2_score(y_test_original, y_pred_original)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mae_str = f"{mae:.4f}".replace('.', '_')
            model_id_base = (
                f"{timestamp}_{mae_str}_RF_{n_estimators}_trees_{max_depth}_depth_"
                f"{min_samples_split}_split_{min_samples_leaf}_leaf_{bootstrap}_bootstrap_"
                f"{max_features}_maxfeat_{max_samples}_maxsamp_{min_weight_fraction_leaf}_minwfrac_"
                f"{oob_score}_oob_{ccp_alpha}_ccp_{criterion}_crit_{args.dataset}"
            )

            # File names
            preds_csv_name = f"{model_id_base}.csv"
            preds_csv_path = os.path.join(log_dir, preds_csv_name)
            model_save_name = f"{model_id_base}.pkl"
            model_save_path = os.path.join(trained_models_dir, model_save_name)

            # 1. Save y_true and y_pred as CSV
            df_preds = pd.DataFrame({
                "y_true": y_test_original.flatten(),
                "y_pred": y_pred_original.flatten()
            })
            df_preds.to_csv(preds_csv_path, index=False)
            # 2. Save the trained model
            joblib.dump(model, model_save_path)
            # 3. Log
            log_file.write(
                f"Model=RF, id={model_id_base}, dataset={args.dataset}, "
                f"n_estimators={n_estimators}, max_depth={max_depth}, "
                f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, "
                f"bootstrap={bootstrap}, max_features={max_features}, max_samples={max_samples}, "
                f"min_weight_fraction_leaf={min_weight_fraction_leaf}, oob_score={oob_score}, "
                f"ccp_alpha={ccp_alpha}, criterion={criterion}, "
                f"MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2_test:.4f}\n"
            )
            log_file.flush()  # Ensure log writes immediately to disk

    print("Training complete.")

if __name__ == "__main__":
    main()
