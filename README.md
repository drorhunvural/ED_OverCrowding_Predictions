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
  This project develops a deep learning approach to predict Emergency Department (ED) patient flow metrics, such as waiting count, using a complete machine learning pipeline from data preparation to evaluation.
</p>

<h2>Project Structure</h2>
<ol>
  <li>
    <b>Data Preparation</b><br>
    <code>data_preparation.py</code><br>
    <span>
      Processes and integrates data from various sources to generate structured datasets for model training and evaluation.
    </span>
  </li>
  <li>
    <b>Training</b>
    <ul>
      <li>
        <code>train_tsai.py</code> — Train deep learning models from the TSAI library, including:
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
        <code>train_RNNbased.py</code> — Train RNN-based models:
        <ul>
          <li>BiLSTM</li>
          <li>Seq2SeqLSTM</li>
          <li>VanillaLSTM</li>
        </ul>
      </li>
      <li>
        <code>train_randomforest.py</code> — Train a Random Forest model.
      </li>
      <li>
        <code>train_xgboost.py</code> — Train an XGBoost model.
      </li>
    </ul>
  </li>
  <li>
    <b>Evaluation</b><br>
    <code>evaluate_tsai.py</code><br>
    <span>
      Evaluate trained models on test data using standard metrics and extreme-case analysis.
    </span>
  </li>
  <li>
    <b>Prediction</b><br>
    <code>predict_tsai.py</code><br>
    <span>
      Generate predictions using trained models for new or unseen data.
    </span>
  </li>
</ol>

<h2>Data Preparation</h2>
<p>
  To prepare your dataset, run the following command:
</p>
<pre><code>python data_preparation.py</code></pre>
<p>
  The script will process raw data, apply feature engineering, and generate structured datasets for model training and evaluation.<br>
  Make sure to update your configuration settings in <code>config/dataset_config.json</code> before running the script.
</p>

<h2>Requirements</h2>
<p>
  Before running any scripts, please install the required dependencies:
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
  Additional dependencies may be needed based on your environment (such as <code>joblib</code> or <code>seaborn</code>).
</p>

<h2>Quick Start</h2>
<ul>
  <li>Prepare the data: <code>python data_preparation.py</code></li>
  <li>Train a model: <code>python train_tsai.py</code></li>
  <li>Evaluate a model: <code>python evaluate_tsai.py</code></li>
  <li>Make predictions: <code>python predict_tsai.py</code></li>
</ul>

<h2>Citation</h2>
<p>
  If you use this code or find this project helpful in your research, please cite our work.<br>
  <b>Stay tuned—our paper is currently under review and will be available soon.</b>
</p>

</body>
</html>
