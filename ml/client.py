from forecasting_framework import ForecastFramework
from forecasting_models import ClosePriceFM


if __name__ == "__main__":

    # Option 1:
    #
    # Create a new framework object
    # fm = ForecastFramework(target_columns=['Close'], forecast_model=ClosePriceFM(), name='xgb_model')

    # Train model
    # fm.train_model()

    # Dump model
    # path: str = "xgb_model"
    # fm.dump_model(path=path)

    # Option 2:
    #
    # Load existing model
    path: str = "xgb_model"
    fm = ForecastFramework.load_from_file(path=path,
                                          target_columns=['Close'],
                                          forecast_model=ClosePriceFM()
                                          )

    # Create forecast
    unit = 'h'  # units of time (e.g. 'h' for hour, 'd' for days, 'm' for months)
    value = 24   # value of units

    # Obtain pandas series with forecasted data
    forecast = fm.create_forecast(value=value, unit=unit)
