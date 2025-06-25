import pandas as pd
import numpy as np
import os
from forecasting_models import LSTMCloseFM, ClosePriceFM, VolumeFM
from forecasting_framework import ForecastFramework

def load_data():
    """Load the gold futures data with indicators"""
    data_path = "data/gold_futures_with_indicators.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found at {data_path}")
        print("Please ensure you have the gold_futures_with_indicators.csv file in the data/ directory")
        return None
    
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"✅ Loaded data shape: {df.shape}")
    print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
    print(f"📊 Columns: {list(df.columns)}")
    
    return df

def train_mvp_lstm_model(df):
    """Train the MVP LSTM model"""
    print("\n🚀 === Training MVP LSTM Model ===")
    
    # Initialize the MVP LSTM model
    lstm_model = LSTMCloseFM(lag=24, prediction_mode='original')
    
    # Train the model
    print("🔄 Training LSTM model...")
    print("⚠️  This may take a while depending on your server capacity...")
    
    try:
        lstm_model.fit(df, epochs=50, patience=10)
        print("✅ LSTM model training completed!")
        
        # Save the model
        model_path = "baseline_model"
        os.makedirs(model_path, exist_ok=True)
        lstm_model.dump(model_path)
        print(f"💾 LSTM model saved to {model_path}/")
        
        return lstm_model
        
    except Exception as e:
        print(f"❌ Error training LSTM model: {e}")
        print("🔄 Falling back to basic XGBoost model...")
        return train_basic_model(df)

def train_basic_model(df):
    """Train the basic XGBoost model as fallback"""
    print("\n🔄 === Training Basic XGBoost Model ===")
    
    # Initialize the basic model
    basic_model = ClosePriceFM(lag=25)
    
    # Train the model
    print("🔄 Training basic model...")
    basic_model.fit(df)
    
    # Save the model
    model_path = "baseline_model"
    os.makedirs(model_path, exist_ok=True)
    basic_model.dump(model_path)
    print(f"💾 Basic model saved to {model_path}/")
    
    return basic_model

def test_model(model, df, model_name):
    """Test the trained model with predictions"""
    print(f"\n🧪 === Testing {model_name} ===")
    
    # Create a date range for the next 24 hours
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), 
                                periods=24, freq='H')
    
    try:
        # Make predictions
        predictions = model.predict(future_dates)
        
        print(f"📈 Predicted values for next 24 hours:")
        for i, (date, pred) in enumerate(zip(future_dates, predictions)):
            # Handle both numpy arrays and scalar values
            pred_value = float(pred) if hasattr(pred, '__iter__') else pred
            print(f"  {i+1:2d}h: {date.strftime('%Y-%m-%d %H:%M')} → {pred_value:.4f}")
        
        return predictions
        
    except Exception as e:
        print(f"❌ Error making predictions: {e}")
        return None

def main():
    """Main training function"""
    print("🎯 === MVP Gold Price Forecasting Model Training ===")
    print("📝 Your teammate uploaded the MVP model code. Training on server...")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Check server capacity (basic check)
    import psutil
    memory_gb = psutil.virtual_memory().total / (1024**3)
    print(f"💻 Server memory: {memory_gb:.1f} GB")
    
    if memory_gb < 4:
        print("⚠️  Low memory detected. Using basic model.")
        model = train_basic_model(df)
        model_name = "Basic XGBoost"
    else:
        print("✅ Sufficient memory for LSTM training.")
        model = train_mvp_lstm_model(df)
        model_name = "MVP LSTM" if isinstance(model, LSTMCloseFM) else "Basic XGBoost"
    
    # Test the model
    test_model(model, df, model_name)
    
    print(f"\n🎉 === Training Complete ===")
    print(f"✅ {model_name} model has been trained and saved!")
    print("📁 Models saved to 'baseline_model/' directory")
    print("🌐 You can now use these models in your web API")

if __name__ == "__main__":
    main() 