<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ED Overcrowding Predictions</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; line-height: 1.6; color: #24292e; max-width: 800px; margin: 0 auto; padding: 2rem; }
    h1 { border-bottom: 1px solid #eaecef; padding-bottom: .3em; }
    h2 { border-bottom: 1px solid #eaecef; padding-bottom: .3em; margin-top: 24px; }
    code { background-color: rgba(27,31,35,.05); border-radius: 3px; font-size: 85%; margin: 0; padding: .2em .4em; font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; }
    pre { background-color: #f6f8fa; border-radius: 3px; padding: 16px; overflow: auto; line-height: 1.45; }
    pre code { background-color: transparent; padding: 0; }
    ul, ol { padding-left: 2em; }
    li { margin-bottom: 0.5em; }
    .note { background-color: #fffbdd; border: 1px solid #d1d5da; padding: 10px; border-radius: 6px; }
  </style>
</head>
<body>

<h1>ED Overcrowding Predictions</h1>

<section>
  <h2>Project Overview</h2>
  <p>
    This project presents a deep learning–based framework for predicting Emergency Department (ED)
    patient flow metrics, including waiting counts and overcrowding-related indicators.
    The repository provides a complete machine learning pipeline covering data preparation,
    model training, evaluation, and prediction.
  </p>
</section>

<section>
  <h2>Project Structure</h2>
  <ol>
    <li>
      <strong>Data Preparation</strong><br>
      <code>data_preparation.py</code>
      <p>Processes and integrates data from multiple sources to generate structured datasets for model training and evaluation.</p>
      <p class="note">
        <strong>Note:</strong> All data sources are expected to be placed in the <code>data_source</code> folder. 
        Due to institutional data privacy restrictions, original hospital data is not included. 
        Synthetic data is provided for demonstration purposes.
      </p>
    </li>

    <li>
      <strong>Training Scripts</strong>
      <ul>
        <li><code>train_tsai.py</code>: Deep learning time-series models (TSiTPlus, TSTPlus, FCNPlus, RNNPlus, etc.)</li>
        <li><code>train_RNNbased.py</code>: Recurrent neural network models (BiLSTM, Seq2Seq LSTM, Vanilla LSTM)</li>
        <li><code>train_randomforest.py</code>: Random Forest regression</li>
        <li><code>train_xgboost.py</code>: XGBoost regression</li>
      </ul>
    </li>

    <li>
      <strong>Evaluation</strong><br>
      <code>evaluate_tsai.py</code>
      <p>Evaluates trained models using standard performance metrics and extreme-case analyses.</p>
    </li>

    <li>
      <strong>Prediction</strong><br>
      <code>predict_tsai.py</code>
      <p>Generates predictions for new or unseen data using trained models.</p>
    </li>
  </ol>
</section>

<section>
  <h2>Requirements</h2>
  <pre><code>pip install tsai optuna torch scikit-learn matplotlib pandas numpy</code></pre>
</section>

<section>
  <h2>Quick Start</h2>
  <ol>
    <li><strong>Prepare data:</strong> <code>python data_preparation.py</code></li>
    <li><strong>Train model:</strong> <code>python train_tsai.py</code></li>
    <li><strong>Evaluate:</strong> <code>python evaluate_tsai.py</code></li>
    <li><strong>Predict:</strong> <code>python predict_tsai.py</code></li>
  </ol>
</section>

<section>
  <h2>Citation</h2>
  <blockquote>
    Vural, O., Ozaydin, B., Aram, K. Y., Booth, J., Lindsey, B. F., &amp; Ahmed, A. (2025). 
    <em>An Artificial Intelligence–Based Framework for Predicting Emergency Department Overcrowding: Development and Evaluation Study.</em> 
    <strong>JMIR Medical Informatics</strong>, 13, e73960.
  </blockquote>
</section>

</body>
</html>
