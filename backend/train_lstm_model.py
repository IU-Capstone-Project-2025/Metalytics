import os
import pandas as pd
import numpy as np
import json
import subprocess
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from forecasting_framework import ForecastFramework
from forecasting_models import LSTMCloseFM

# Paths
ML_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../ml'))
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
DATA_PATH = os.path.join(DATA_DIR, 'gold_futures_with_indicators.csv')
LSTM_MODEL_DIR = os.path.join('lstm_model')
MODEL_FILE = os.path.join(LSTM_MODEL_DIR, 'LSTM_Close_model.keras')
SCALER_FILE = os.path.join(LSTM_MODEL_DIR, 'LSTM_Close_scaler.pkl')
METRICS_FILE = os.path.join(LSTM_MODEL_DIR, 'metrics_lstm.json')

os.makedirs(LSTM_MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_and_prepare_data():
    # Run fetch_gold_ohlcv.py (it will output to ml/ by default)
    subprocess.run(['python', os.path.join(ML_DIR, 'fetch_gold_ohlcv.py')], check=True)
    # Run calculate_indicators.py, then move the output to backend/data/
    subprocess.run(['python', os.path.join(ML_DIR, 'calculate_indicators.py')], check=True)
    src = os.path.join(ML_DIR, 'gold_futures_with_indicators.csv')
    if os.path.exists(src):
        os.replace(src, DATA_PATH)
    print('Data preparation complete.')

def train_and_evaluate_lstm():
    fetch_and_prepare_data()
    print('Loading data...')
    df = pd.read_csv(DATA_PATH, parse_dates=[0], index_col=0)

    print('Initializing LSTM model...')
    model = LSTMCloseFM()
    # You can tune params here if needed
    model.fit(df)

    print('Saving model and scaler...')
    model.dump(LSTM_MODEL_DIR)

    # Evaluate on the last 24 hours (or as appropriate)
    print('Evaluating model...')
    forecast_hours = 24
    date_range = pd.date_range(df.index[-1] + pd.Timedelta(hours=1), periods=forecast_hours, freq='h')
    predictions = model.predict(date_range)

    # For metrics, compare to actuals if available (here, just as a placeholder)
    # In real use, you should have a test set with actual future values
    # Here, we'll use the last 24 actuals as a proxy
    if len(df) >= forecast_hours:
        actuals = df['Close'].iloc[-forecast_hours:]
        # Align indices if needed
        actuals = actuals.reset_index(drop=True)
        preds = predictions.reset_index(drop=True)
        mae = mean_absolute_error(actuals, preds)
        mse = mean_squared_error(actuals, preds)
        r2 = r2_score(actuals, preds)
    else:
        mae = mse = r2 = None

    metrics = {
        'MAE': mae,
        'MSE': mse,
        'R2': r2,
        'info': 'Metrics are calculated using the last 24 actual closes as a proxy.'
    }
    print('Metrics:', metrics)
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'Metrics saved to {METRICS_FILE}')

if __name__ == '__main__':
    train_and_evaluate_lstm() 