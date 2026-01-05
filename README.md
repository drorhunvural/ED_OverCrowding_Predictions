<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ED Overcrowding Predictions</title>
</head>
<body style="font-family: sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: auto; padding: 20px;">

    <h1 style="border-bottom: 2px solid #eee; padding-bottom: 10px;">ED Overcrowding Predictions</h1>

    <h2 style="color: #2c3e50;">Project Overview</h2>
    <p>
        This project presents a deep learning–based framework for predicting Emergency Department (ED)
        patient flow metrics, including waiting counts and overcrowding-related indicators.
    </p>

    <h2 style="color: #2c3e50;">Project Structure</h2>
    <ul style="list-style-type: none; padding-left: 0;">
        <li style="margin-bottom: 20px;">
            <strong>1. Data Preparation</strong> (<code>data_preparation.py</code>)
            <p style="margin-top: 5px;">Processes data sources to generate structured datasets.</p>
            <div style="background: #fffbdd; border: 1px solid #d1d5da; padding: 10px; font-size: 0.9em;">
                <strong>Note:</strong> Data must be in the <code>data_source</code> folder. 
                Synthetic data is provided for demonstration.
            </div>
        </li>

        <li style="margin-bottom: 20px;">
            <strong>2. Training Scripts</strong>
            <ul>
                <li><code>train_tsai.py</code>: Deep learning time-series (TSiTPlus, TSTPlus, etc.)</li>
                <li><code>train_RNNbased.py</code>: RNN models (BiLSTM, Seq2Seq LSTM)</li>
                <li><code>train_randomforest.py</code>: Random Forest regression</li>
                <li><code>train_xgboost.py</code>: XGBoost regression</li>
            </ul>
        </li>

        <li style="margin-bottom: 20px;">
            <strong>3. Evaluation & Prediction</strong>
            <p>Use <code>evaluate_tsai.py</code> for metrics and <code>predict_tsai.py</code> for new data.</p>
        </li>
    </ul>

    <h2 style="color: #2c3e50;">Requirements</h2>
    <pre style="background: #f4f4f4; border: 1px solid #ddd; padding: 15px; overflow-x: auto;">pip install tsai optuna torch scikit-learn matplotlib pandas numpy</pre>

    <h2 style="color: #2c3e50;">Quick Start</h2>
    <ol>
        <li>Prepare: <code>python data_preparation.py</code></li>
        <li>Train: <code>python train_tsai.py</code></li>
        <li>Evaluate: <code>python evaluate_tsai.py</code></li>
    </ol>

    <h2 style="color: #2c3e50;">Citation</h2>
    <p style="font-style: italic; background: #f9f9f9; padding: 15px; border-left: 5px solid #2c3e50;">
        Vural, O., et al. (2025). An Artificial Intelligence–Based Framework for Predicting Emergency Department Overcrowding. <strong>JMIR Medical Informatics</strong>, 13, e73960.
    </p>

</body>
</html>
