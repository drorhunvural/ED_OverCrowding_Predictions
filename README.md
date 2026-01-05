<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ED_OverCrowding_Predictions</title>
</head>
<body>

<h1>ED_OverCrowding_Predictions</h1>

<h2>Project Overview</h2>
<p>
  This project presents a deep learning–based framework for predicting Emergency Department (ED) patient flow metrics,
  including waiting counts and related overcrowding indicators. The repository provides a complete machine learning
  pipeline, covering data preparation, model training, evaluation, and prediction.
</p>

<h2>Project Structure</h2>
<ol>
  <li>
    <b>Data Preparation</b><br>
    <code>data_preparation.py</code><br>
    <span>
      Processes and integrates data from multiple sources to generate structured datasets suitable for model training and evaluation.
      <br><br>
      <b>Note:</b> All data sources are expected to be placed in the <code>data_source</code> folder. Due to data privacy
      and institutional restrictions, original hospital data is not included in this repository. Synthetic or dummy data
      is provided for demonstration and reproducibility purposes.
    </span>
  </li>

  <li>
    <b>Training</b>
    <ul>
      <li>
        <code>train_tsai.py</code> — Train deep learning time-series models using the TSAI library, including:
        <ul>
          <li>TSiTPlus</li>
          <li>TSTPlus</li>
          <li>FCNPlus</li>
          <li>RNNPlus</li>
          <li>ResNetPlus</li>
          <li>XCMPlus</li>
          <li>XceptionTimePlus</li>
        </ul>
      </li>

      <li>
        <code>train_RNNbased.py</code> — Train recurrent neural network–based models:
        <ul>
          <li>BiLSTM</li>
          <li>Seq2Seq LSTM</li>
          <li>Vanilla LSTM</li>
        </ul>
      </li>

      <li>
        <code>train_randomforest.py</code> — Train a Random Forest regression model.
      </li>

      <li>
        <code>train_xgboost.py</code> — Train an XGBoost regression model.
      </li>
    </ul>
  </li>

  <li>
    <b>Evaluation</b><br>
    <code>evaluate_tsai.py</code><br>
    <span>
      Evaluates trained models using standard performance metrics as well as extreme-case and stress-condition analyses.
    </span>
  </li>

  <li>
    <b>Prediction</b><br>
    <code>predict_tsai.py</code><br>
    <span>
      Generates predictions for new or unseen data using trained and saved models.
    </span>
  </li>
</ol>

<h2>Data Preparation</h2>
<p>
  To prepare the dataset, run:
</p>
<pre><code>python data_preparation.py</code></pre>
<p>
  This script reads data from the <code>data_source</code> directory, applies feature engineering and preprocessing steps,
  and produces structured datasets for training and evaluation.
  <br><br>
  Ensure that the configuration parameters in <code>config/dataset_config.json</code> are properly set before execution.
</p>

<p>
  <b>Important:</b> Due to institutional data access restrictions, original hospital data cannot be shared.
  Dummy data is included solely for demonstration and testing purposes.
</p>

<h2>Requirements</h2>
<p>
  Install the required dependencies before running the scripts:
</p>
<pre><code>pip install tsai
pip install optuna
pip install torch
pip install scikit-learn
pip install matplotlib
pip install pandas
pip install numpy
</code></pre>

<p>
  Additional dependencies such as <code>joblib</code> may be required depending on your environment and configuration.
</p>

<h2>Quick Start</h2>
<ul>
  <li>Prepare the data: <code>python data_preparation.py</code></li>
  <li>Train a model: <code>python train_tsai.py</code></li>
  <li>Evaluate a model: <code>python evaluate_tsai.py</code></li>
  <li>Generate predictions: <code>python predict_tsai.py</code></li>
</ul>

<h2>Citation</h2>
<p>
  If you use this code or find this project helpful in your research, please cite the following paper:
</p>

<p>
  Vural, O., Ozaydin, B., Aram, K. Y., Booth, J., Lindsey, B. F., &amp; Ahmed, A. (2025).
  <i>An Artificial Intelligence–Based Framework for Predicting Emergency Department Overcrowding:
  Development and Evaluation Study.</i>
  <b>JMIR Medical Informatics</b>, 13, e73960.
</p>

</body>
</html>
