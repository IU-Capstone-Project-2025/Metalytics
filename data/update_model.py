from forecasting_framework import ForecastFramework
from forecasting_models import ClosePriceFM


def update_model():
    # Create a new framework object
    fm = ForecastFramework(
        target_columns=['Close'],
        forecast_model=ClosePriceFM(),
        name='xgb_model'
    )

    # Train model
    fm.train_model()

    # Dump model
    path: str = "xgb_model"
    fm.dump_model(path=path)
