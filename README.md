# ED_OverCrowding_Predictions

## Project Overview

This project presents a deep learning–based framework for predicting Emergency Department (ED)
patient flow metrics, including waiting counts and overcrowding-related indicators.
The repository provides a complete machine learning pipeline covering data preparation,
model training, evaluation, and prediction.

---

## Project Structure

### 1. Data Preparation

**Script:** `data_preparation.py`

Processes and integrates data from multiple sources to generate structured datasets
for model training and evaluation.

> **Note:**  
> All data sources are expected to be placed in the `data_source` folder.  
> Due to institutional data privacy restrictions, original hospital data is not included
> in this repository. Synthetic or dummy data is provided for demonstration purposes only.

---

### 2. Training

- **`train_tsai.py`** — Train deep learning time-series models using the TSAI library:
  - TSiTPlus
  - TSTPlus
  - FCNPlus
  - RNNPlus
  - ResNetPlus
  - XCMPlus
  - XceptionTimePlus

- **`train_RNNbased.py`** — Train recurrent neural network–based models:
  - BiLSTM
  - Seq2Seq LSTM
  - Vanilla LSTM

- **`train_randomforest.py`** — Train a Random Forest regression model

- **`train_xgboost.py`** — Train an XGBoost regression model

---

### 3. Evaluation

**Script:** `evaluate_tsai.py`

Evaluates trained models using standard performance metrics and extreme-case analyses.

---

### 4. Prediction

**Script:** `predict_tsai.py`

Generates predictions for new or unseen data using trained and saved models.

---

## Data Preparation

To prepare the dataset, run:

```bash
python data_preparation.py
