import pandas as pd
import os
from forecasting_models import ForecastModel, ClosePriceFM


class ForecastFramework:
    """
    Class for creating, maintaining, dumping, and loading forecasting models.

    Attributes:
        df (pd.DataFrame): dataframe for models.
        forecast_model (ForecastModel): model forecasting target.
    """
    df: pd.DataFrame
    forecast_model: ForecastModel

    def __init__(self, df: pd.DataFrame, forecast_model: ForecastModel = ClosePriceFM(), name="baseline_model"):
        self.df = df
        self.forecast_model = forecast_model
        self.name = name

    def train_model(self) -> None:
        """
        Fits the model with dataframe.
        """
        self.forecast_model.fit(self.df)

    def dump_model(self, path: str = None) -> None:
        """
        Dumps the model to the given path.

        Parameters:
            path (str): path to the folder to store model files.
        """
        if path is None:
            path = self.name
        if not os.path.exists(path):
            os.mkdir(path)
        self.forecast_model.dump(path=path)

    def load_from_file(
            path: str,
            df: pd.DataFrame,
            forecast_model: ForecastModel = ClosePriceFM(),
            name="baseline_model"
    ):
        """
        (Constructor)
        Loads the model from the given path with the dataframe.

        Parameters:
            path (str): path to the folder with model files.
            df (pd.DataFrame): dataframe for model fitting.
            forecast_model (ForecastModel): model forecasting target.
            name (str): name of the model.

        Returns:
            ForecastFramework: constructed framework object.
        """
        assert os.path.exists(path)
        framework = ForecastFramework(df, forecast_model, name)
        framework.forecast_model.load(df, path)
        return framework

    def create_forecast(self, value: int, unit: str) -> pd.Series:
        """
        Predict values from the last observation by value units of time.

        Parameters:
            value (int): number of units.
            unit (str): unit of time (e.g. 'h', 'd', 'm')

        Returns:
            pd.Series: forecasted values.
        """
        date_range = self.forecast_interval_(value, unit)
        return self.forecast_model.predict(date_range)

    def forecast_interval_(self, value: int, unit: str) -> pd.DatetimeIndex:
        """
        Produces date range from the last available observation + 1 hour
        to the date after `value` number of `unit`s.

        Parameters:
            value (int): number of units.
            unit (str): unit of time (e.g. 'h', 'd', 'm')

        Returns:
            pd.DatetimeIndex: date range for forecasting model.
        """
        date_index = self.df.index
        return pd.date_range(
            date_index[-1] + pd.Timedelta(value=1, unit='h'),
            date_index[-1] + pd.Timedelta(value=1, unit='h') + pd.Timedelta(value=value, unit=unit),
            freq='h'
        )
