import pandas as pd
import os
import json
import shutil
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Path Configuration
# -----------------------------
BASE_PATH = "tsai"  # Main project folder
CONFIG_PATH = os.path.join("config")
DATA_SOURCES_PATH = os.path.join("data_sources")
DATASETS_PATH = os.path.join("datasets")
LOGS_PATH = os.path.join(BASE_PATH, "logs")
TRAINED_PATH = os.path.join(BASE_PATH, "trainedfiles")

# Create base directories if they don't exist
for path in [BASE_PATH, CONFIG_PATH, DATA_SOURCES_PATH, DATASETS_PATH, LOGS_PATH, TRAINED_PATH]:
    os.makedirs(path, exist_ok=True)

print(f"Project structure created under: {BASE_PATH}")
print(f"Config path: {CONFIG_PATH}")
print(f"Data sources path: {DATA_SOURCES_PATH}")
print(f"Datasets output path: {DATASETS_PATH}")
print(f"Logs path: {LOGS_PATH}")
print(f"Trained models path: {TRAINED_PATH}")

# -----------------------------
# Helper Functions
# -----------------------------
def round_and_cast_to_int(df):
    for col in df.select_dtypes(include=['float', 'int']).columns:
        df[col] = df[col].round()
    return df

def filter_range(df, start_time, end_time):
    return df[(df['hourly_range'] >= start_time) & (df['hourly_range'] <= end_time)].reset_index(drop=True)

def extreme_indicator(df, column_name):
    mean_value = df[column_name].mean()
    std_dev = df[column_name].std()
    threshold = mean_value + std_dev
    df['extreme_indicator'] = df[column_name].apply(lambda x: 1 if x >= threshold else 0)
    return df

def categorize_weather(weather):
    if weather in ['Clouds', 'Mist']:
        return 'Clouds'
    elif weather in ['Rain', 'Drizzle']:
        return 'Rain'
    elif weather == 'Thunderstorm':
        return 'Thunderstorm'
    elif weather == 'Clear':
        return 'Clear'
    else:
        return 'Others'

def generate_lagged_and_rolling_features(N, future_hours, df, window_size, column_name):
    df = df.copy()
    for i in range(1, N + 1):
        df[f'lag_{i}_{column_name}'] = df[column_name].shift(i)
    for i in range(1, future_hours + 1):
        df[f'future_{i}_{column_name}'] = df[column_name].shift(-i)
    if window_size > 0:
        df[f'rolling_mean_{column_name}_window_size_{window_size}'] = df[column_name].rolling(window=window_size).mean()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    drop_cols = [f'future_{i}_{column_name}' for i in range(1, future_hours)]
    df.drop(columns=drop_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# -----------------------------
# Load config from JSON
# -----------------------------
config_file_path = os.path.join(CONFIG_PATH, "dataset_config.json")
if not os.path.exists(config_file_path):
    raise FileNotFoundError(f"Configuration file not found: {config_file_path}")

with open(config_file_path, "r") as f:
    config = json.load(f)

dataset_number = config["dataset_number"]
desired_features = config["features"]
covid_start = config["covid_filter_start"]
covid_end = config["covid_filter_end"]
target_feature = config["target_feature"]
covid_deletion = config.get("covid_deletion", False)
N = config["lag_N"]
future_hours = config["future_hours"]
window_size = config.get("window_size", 0)
train_fraction = config.get("train_fraction", 0.7)
val_fraction = config.get("validation_fraction", 0.15)

if target_feature not in desired_features:
    raise ValueError(f"target_feature '{target_feature}' must be included in the 'features' list.")

print(f"Loading configuration for dataset: {dataset_number}")
print(f"Target feature: {target_feature}")
print(f"Number of lag features: {N}")
print(f"Future hours to predict: {future_hours}")

# -----------------------------
# Load and preprocess input data
# -----------------------------
start_time = '2018-12-31 12:00:00'
end_time = '2023-07-01 20:00:00'

# Helper function to read CSV files from data sources
read_csv = lambda filename: pd.read_csv(os.path.join(DATA_SOURCES_PATH, filename))

print(f"Loading data files from: {DATA_SOURCES_PATH}")

# Check if required files exist
required_files = [
    'df_UED_Total_Patient_Count_Average_Waiting_Time_ESI_Weather_39440_V3.csv',
    'df_UED_Hospital_Census_V1.csv',
    'df_UED_Treatment_Count_Average_Treatment_Time_V3.csv',
    'df_UED_Boarding_Count_Average_Boarding_Time_V8.csv',
    'df_UED_Alabama_Crimson_football_game_V1.csv',
    'df_UED_auburn_football_game_V1.csv',
    'federalholidays_2019_2023_V1.csv'
]

missing_files = []
for file in required_files:
    if not os.path.exists(os.path.join(DATA_SOURCES_PATH, file)):
        missing_files.append(file)

if missing_files:
    print(f"WARNING: Missing data files in {DATA_SOURCES_PATH}:")
    for file in missing_files:
        print(f"  - {file}")
    print("Please ensure all required data files are present before running.")
    
# Load data files (assuming they exist)
try:
    df_patient_average_count = read_csv('df_UED_Total_Patient_Count_Average_Waiting_Time_ESI_Weather_39440_V3.csv')
    df_UED_hospital_census = read_csv('df_UED_Hospital_Census_V1.csv')
    df_treatment_average_count = read_csv('df_UED_Treatment_Count_Average_Treatment_Time_V3.csv')
    df_boarding = read_csv('df_UED_Boarding_Count_Average_Boarding_Time_V8.csv')
    df_alabama_game = read_csv('df_UED_Alabama_Crimson_football_game_V1.csv')
    df_auburn_game = read_csv('df_UED_auburn_football_game_V1.csv')
    df_holidays = read_csv('federalholidays_2019_2023_V1.csv')
    print("✅ All data files loaded successfully")
except Exception as e:
    print(f"❌ Error loading data files: {str(e)}")
    raise

# Rename and format
df_UED_hospital_census.rename(columns={'UED Hospital Census': 'UED_Hospital_Census', 'DateTime': 'hourly_range'}, inplace=True)
df_boarding.rename(columns={'DateTime': 'hourly_range'}, inplace=True)

# Parse datetime
for df in [df_patient_average_count, df_UED_hospital_census, df_treatment_average_count, df_boarding, df_alabama_game, df_auburn_game, df_holidays]:
    df['hourly_range'] = pd.to_datetime(df['hourly_range'])
    df.sort_values('hourly_range', inplace=True)

# Filter date range
df_patient_average_count = filter_range(df_patient_average_count, start_time, end_time)
df_UED_hospital_census = filter_range(df_UED_hospital_census, start_time, end_time)
df_treatment_average_count = filter_range(df_treatment_average_count, start_time, end_time)
df_boarding = filter_range(df_boarding, start_time, end_time)
df_holidays = filter_range(df_holidays, start_time, end_time)
df_auburn_game = filter_range(df_auburn_game, start_time, end_time)
df_alabama_game = filter_range(df_alabama_game, start_time, end_time)

# Drop unnecessary columns
for df in [df_treatment_average_count, df_boarding]:
    df.drop(columns=['patient_ids', 'patient_visit_ids'], inplace=True, errors='ignore')

# Round numerics (leave as float for scaling)
for df in [df_patient_average_count, df_UED_hospital_census, df_treatment_average_count, df_boarding, df_holidays, df_auburn_game, df_alabama_game]:
    round_and_cast_to_int(df)

print("✅ Data preprocessing completed")

# Merge and feature engineering
print("🔄 Merging datasets and creating features...")
df_merged = pd.merge(df_UED_hospital_census, df_patient_average_count, on='hourly_range', how='inner')
df_merged = pd.merge(df_merged, df_boarding, on='hourly_range', how='inner')
df_merged = pd.merge(df_merged, df_treatment_average_count, on='hourly_range', how='inner')
df_merged = df_merged.merge(df_auburn_game[['hourly_range', 'Auburn_Football_Game_Actual_Time']], on='hourly_range', how='left')
df_merged['Auburn_Football_Game_Actual_Time'] = df_merged['Auburn_Football_Game_Actual_Time'].fillna(0).astype(int)
df_merged = df_merged.merge(df_alabama_game[['hourly_range', 'Alabama_Football_Game_Actual_Time']], on='hourly_range', how='left')
df_merged['Alabama_Football_Game_Actual_Time'] = df_merged['Alabama_Football_Game_Actual_Time'].fillna(0).astype(int)

# Merge holidays
df_merged['date'] = df_merged['hourly_range'].dt.date
df_holidays['date'] = df_holidays['hourly_range'].dt.date
df_merged = pd.merge(df_merged, df_holidays[['date', 'Federal_Holiday']], on='date', how='left')
df_merged['Federal_Holiday'] = df_merged['Federal_Holiday'].fillna(0).astype(int)
df_merged.drop(columns=['date'], inplace=True)

# Weather processing
df_merged['weather_main'] = df_merged['weather_main'].apply(categorize_weather)
df_merged.rename(columns={'weather_main': 'weather_type'}, inplace=True)
df_encoded = pd.get_dummies(df_merged['weather_type'], prefix='weather').astype(int)
df_merged = pd.concat([df_merged, df_encoded], axis=1)

# Time-based features
df_merged['year'] = df_merged['hourly_range'].dt.year
df_merged['month'] = df_merged['hourly_range'].dt.month
df_merged['day_of_month'] = df_merged['hourly_range'].dt.day
df_merged['hour'] = df_merged['hourly_range'].dt.hour
df_merged['day_of_week'] = df_merged['hourly_range'].dt.weekday

if covid_deletion:
    print(f"🦠 Removing COVID period: {covid_start} to {covid_end}")
    df_merged = df_merged[~((df_merged['hourly_range'] >= covid_start) & (df_merged['hourly_range'] <= covid_end))]
    df_merged.reset_index(drop=True, inplace=True)

# Add extreme indicator
df_merged = extreme_indicator(df_merged, target_feature)

# Drop unneeded columns
df_merged.drop(columns=['weather_type', 'hourly_range', 'weather_description'], inplace=True, errors='ignore')

# Create output directory under DATASETS_PATH with future_hours folder structure
future_hours_folder = f"{future_hours}hours"
output_dir = os.path.join(DATASETS_PATH, future_hours_folder, dataset_number)
os.makedirs(output_dir, exist_ok=True)
print(f"📁 Created dataset directory: {output_dir}")
print(f"⏰ Future hours: {future_hours} → Folder: {future_hours_folder}")

# Save full merged dataset before dropping columns
df_merged.to_csv(os.path.join(output_dir, f'df_merged_{dataset_number}.csv'), index=False)
print(f"💾 Saved merged dataset: {os.path.join(output_dir, f'df_merged_{dataset_number}.csv')}")

# Keep selected features + target
df_merged = df_merged[desired_features]

# Generate lag and rolling features
print(f"🔄 Generating {N} lag features and predicting {future_hours} hours ahead...")
df_merged = generate_lagged_and_rolling_features(N, future_hours, df_merged, window_size, target_feature)
actual_target_col_name = f'future_{future_hours}_{target_feature}'

print(f"📊 Final dataset shape: {df_merged.shape}")
print(f"🎯 Target column: {actual_target_col_name}")

# Split dataset
total_samples = len(df_merged)
train_end = int(total_samples * train_fraction)
val_end = train_end + int(total_samples * val_fraction)
train_df = df_merged.iloc[:train_end].copy()
val_df = df_merged.iloc[train_end:val_end].copy()
test_df = df_merged.iloc[val_end:].copy()

X_train_unscaled = train_df.drop(columns=[actual_target_col_name])
y_train_unscaled = train_df[[actual_target_col_name]]
X_val_unscaled = val_df.drop(columns=[actual_target_col_name])
y_val_unscaled = val_df[[actual_target_col_name]]
X_test_unscaled = test_df.drop(columns=[actual_target_col_name])
y_test_unscaled = test_df[[actual_target_col_name]]

print(f"✂️ Dataset split:")
print(f"  Train: {X_train_unscaled.shape[0]} samples")
print(f"  Validation: {X_val_unscaled.shape[0]} samples") 
print(f"  Test: {X_test_unscaled.shape[0]} samples")

# ---- Save unscaled splits ----
print("💾 Saving unscaled data splits...")
X_train_unscaled.to_csv(os.path.join(output_dir, f'X_train_unscaled_{dataset_number}.csv'), index=False)
X_val_unscaled.to_csv(os.path.join(output_dir, f'X_val_unscaled_{dataset_number}.csv'), index=False)
X_test_unscaled.to_csv(os.path.join(output_dir, f'X_test_unscaled_{dataset_number}.csv'), index=False)
y_train_unscaled.to_csv(os.path.join(output_dir, f'y_train_unscaled_{dataset_number}.csv'), index=False)
y_val_unscaled.to_csv(os.path.join(output_dir, f'y_val_unscaled_{dataset_number}.csv'), index=False)
y_test_unscaled.to_csv(os.path.join(output_dir, f'y_test_unscaled_{dataset_number}.csv'), index=False)

# Save config as record
config_backup_path = os.path.join(output_dir, f"config_{dataset_number}.json")
shutil.copy(config_file_path, config_backup_path)
print(f"📋 Saved config backup: {config_backup_path}")

# ---- Feature Scaling (X and y) ----
print("🔧 Applying feature scaling...")
feature_scaler = StandardScaler()
feature_columns_list = X_train_unscaled.columns.tolist()
columns_to_scale_features = [col for col in feature_columns_list if X_train_unscaled[col].nunique(dropna=False) > 2]

X_train_scaled = X_train_unscaled.astype(float).copy()
X_val_scaled = X_val_unscaled.astype(float).copy()
X_test_scaled = X_test_unscaled.astype(float).copy()

if columns_to_scale_features:
    print(f"   📈 Features to be scaled (non-binary): {len(columns_to_scale_features)} features")
    print(f"   Features: {columns_to_scale_features}")
    X_train_scaled.loc[:, columns_to_scale_features] = feature_scaler.fit_transform(X_train_unscaled[columns_to_scale_features].astype(float))
    X_val_scaled.loc[:, columns_to_scale_features] = feature_scaler.transform(X_val_unscaled[columns_to_scale_features].astype(float))
    X_test_scaled.loc[:, columns_to_scale_features] = feature_scaler.transform(X_test_unscaled[columns_to_scale_features].astype(float))
else:
    print("   ⚠️ No non-binary features found for scaling.")

X_train_scaled.to_csv(os.path.join(output_dir, f'X_train_scaled_{dataset_number}.csv'), index=False)
X_val_scaled.to_csv(os.path.join(output_dir, f'X_val_scaled_{dataset_number}.csv'), index=False)
X_test_scaled.to_csv(os.path.join(output_dir, f'X_test_scaled_{dataset_number}.csv'), index=False)
joblib.dump(feature_scaler, os.path.join(output_dir, f'feature_scaler_{dataset_number}.pkl'))
print(f"💾 Saved feature scaler: {os.path.join(output_dir, f'feature_scaler_{dataset_number}.pkl')}")

# ---- Target Scaling ----
print("🎯 Applying target scaling...")
target_scaler = StandardScaler()
y_train_scaled = y_train_unscaled.astype(float).copy()
y_val_scaled = y_val_unscaled.astype(float).copy()
y_test_scaled = y_test_unscaled.astype(float).copy()

y_train_scaled.loc[:, [actual_target_col_name]] = target_scaler.fit_transform(y_train_unscaled[[actual_target_col_name]].astype(float))
y_val_scaled.loc[:, [actual_target_col_name]] = target_scaler.transform(y_val_unscaled[[actual_target_col_name]].astype(float))
y_test_scaled.loc[:, [actual_target_col_name]] = target_scaler.transform(y_test_unscaled[[actual_target_col_name]].astype(float))

y_train_scaled.to_csv(os.path.join(output_dir, f'y_train_scaled_{dataset_number}.csv'), index=False)
y_val_scaled.to_csv(os.path.join(output_dir, f'y_val_scaled_{dataset_number}.csv'), index=False)
y_test_scaled.to_csv(os.path.join(output_dir, f'y_test_scaled_{dataset_number}.csv'), index=False)
joblib.dump(target_scaler, os.path.join(output_dir, f'target_scaler_{dataset_number}.pkl'))
print(f"💾 Saved target scaler: {os.path.join(output_dir, f'target_scaler_{dataset_number}.pkl')}")

# ---- Save npy 3D/2D for DL ----
print("🤖 Saving 3D numpy arrays for deep learning...")
def save_as_npy(X, y, prefix):
    X_np = X.values.astype(np.float32)
    X_np = np.expand_dims(X_np, axis=-1)  # (n_samples, n_features, 1)
    y_np = y.values.astype(np.float32)    # (n_samples, 1)
    np.save(os.path.join(output_dir, f'{prefix}_X_{dataset_number}.npy'), X_np)
    np.save(os.path.join(output_dir, f'{prefix}_y_{dataset_number}.npy'), y_np)
    print(f"   💾 Saved {prefix}_X_{dataset_number}.npy: {X_np.shape}")
    print(f"   💾 Saved {prefix}_y_{dataset_number}.npy: {y_np.shape}")

save_as_npy(X_train_scaled, y_train_scaled, 'train')
save_as_npy(X_val_scaled, y_val_scaled, 'val')
save_as_npy(X_test_scaled, y_test_scaled, 'test')

# ---- Save explanation and feature order ----
explanation_path = os.path.join(output_dir, f'explanation_{dataset_number}.txt')
print(f"📝 Creating explanation file: {explanation_path}")

with open(explanation_path, 'w') as f:
    f.write(f"Dataset: {dataset_number}\n")
    f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Base path: {BASE_PATH}\n")
    f.write(f"Config path: {CONFIG_PATH}\n")
    f.write(f"Data sources path: {DATA_SOURCES_PATH}\n")
    f.write(f"Output path: {output_dir}\n\n")
    f.write(f"Unscaled and scaled versions of X and y are saved for train/val/test, and 3D npy arrays for DL.\n")
    f.write(f"Features scaled: {columns_to_scale_features}\n")
    f.write(f"Scaler used: StandardScaler (feature and target)\n")
    f.write(f"X_train shape: {X_train_unscaled.shape}, X_val shape: {X_val_unscaled.shape}, X_test shape: {X_test_unscaled.shape}\n")
    f.write(f"y_train shape: {y_train_unscaled.shape}, y_val shape: {y_val_unscaled.shape}, y_test shape: {y_test_unscaled.shape}\n")
    # Add npy shapes
    X_train_np_shape = (X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    y_train_np_shape = (y_train_scaled.shape[0], 1)
    X_val_np_shape = (X_val_scaled.shape[0], X_val_scaled.shape[1], 1)
    y_val_np_shape = (y_val_scaled.shape[0], 1)
    X_test_np_shape = (X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    y_test_np_shape = (y_test_scaled.shape[0], 1)
    f.write(f"\nNumpy array shapes used for deep learning:\n")
    f.write(f"  X_train_np shape: {X_train_np_shape}, y_train_np shape: {y_train_np_shape}\n")
    f.write(f"  X_val_np shape: {X_val_np_shape}, y_val_np shape: {y_val_np_shape}\n")
    f.write(f"  X_test_np shape: {X_test_np_shape}, y_test_np shape: {y_test_np_shape}\n")
    f.write(f"Target column: {actual_target_col_name}\n")
    f.write(f"Feature order used (for model input):\n{feature_columns_list}\n")

print("\n" + "="*60)
print(f"✅ {dataset_number} SUCCESSFULLY GENERATED!")
print("="*60)
print(f"📁 Dataset location: {output_dir}")
print(f"📊 Total samples: {total_samples}")
print(f"🎯 Target: {actual_target_col_name}")
print(f"📈 Features: {len(feature_columns_list)}")
print(f"🔧 Scaled features: {len(columns_to_scale_features)}")
print("\n📋 Generated files:")
files_created = [
    f"✓ {f}_X_{dataset_number}.npy & {f}_y_{dataset_number}.npy" for f in ['train', 'val', 'test']
] + [
    f"✓ X_{split}_scaled_{dataset_number}.csv & y_{split}_scaled_{dataset_number}.csv" for split in ['train', 'val', 'test']
] + [
    f"✓ feature_scaler_{dataset_number}.pkl",
    f"✓ target_scaler_{dataset_number}.pkl", 
    f"✓ config_{dataset_number}.json",
    f"✓ explanation_{dataset_number}.txt"
]

for file_info in files_created:
    print(f"  {file_info}")

print(f"\n🚀 Ready for training with: python train.py --dataset {dataset_number}")
print("="*60)