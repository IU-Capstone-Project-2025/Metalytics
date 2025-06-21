from forecasting_framework import ForecastFramework
import pandas as pd


if __name__ == "__main__":

    # Load dataframe
    dataframe = pd.read_csv("data/gold_futures_with_indicators.csv", parse_dates=[0], index_col=0)

    # Option 1:
    #
    # Create a new framework object
    # fm = ForecastFramework(dataframe)

    # Train model
    # fm.train_model()

    # Dump model
    # path: str = "baseline_model"
    # fm.dump_model()

    # Option 2:
    #
    # Load existing model
    path: str = "baseline_model"
    fm = ForecastFramework.load_from_file(path, dataframe)

    # Create forecast
    unit = 'h'  # units of time (e.g. 'h' for hour, 'd' for days, 'm' for months)
    value = 24   # value of units

    # Obtain pandas series with forecasted data
    forecast = fm.create_forecast(value=value, unit=unit)
