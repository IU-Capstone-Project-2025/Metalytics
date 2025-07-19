from forecasting_framework import ForecastFramework
from forecasting_models import ClosePriceFM, ClosePriceFM_Silver 
from data_loader import GoldDataLoader, SilverDataLoader


# model_list = [
#     'xgb_model',
#     'silver_xgb_model'
# ]

model_params = [
    {
        'data_loader': GoldDataLoader(),
        'target_columns': ['Close'],
        'forecast_model': ClosePriceFM(),
        'name': 'xgb_model'
    },
    {
        'data_loader': SilverDataLoader(),
        'target_columns': ['Close'],
        'forecast_model': ClosePriceFM_Silver(),
        'name': 'silver_xgb_model'
    }
]

def update_model():
    for params in model_params:
        # Create a new framework object
        fm = ForecastFramework(
            data_loader= params['data_loader'],
            target_columns=params['target_columns'],
            forecast_model=params['forecast_model'],
            name=params['name']
        )

        # Train model
        try:
            fm.train_model()
        except Exception as e:
            print(f"Training failed for {params['name']}: {e}")

        # Dump model
        fm.dump_model(path=params['name'])
        print(f"✅ Model '{params['name']}' saved to: {params['name']}/")

if __name__ == "__main__":
    update_model()