from forecasting_framework import ForecastFramework
from forecasting_models import ClosePriceFM


model_list = [
    'xgb_model',
    'silver_xgb_model'
]

def update_model():
    for model in model_list:
        # Create a new framework object
        fm = ForecastFramework(
            target_columns=['Close'],
            forecast_model=ClosePriceFM(),
            name=model
        )

        # Train model
        fm.train_model()

        # Dump model
        path: str = model
        fm.dump_model(path=path)

if __name__ == "__main__":
    update_model()