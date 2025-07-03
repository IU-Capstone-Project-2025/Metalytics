import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
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
    target_columns: pd.DataFrame
    train_set: pd.DataFrame
    test_set: pd.DataFrame
    forecast_model: ForecastModel

    def __init__(
            self,
            data_loader: GoldDataLoader,
            target_columns=['Close'],
            forecast_model=ClosePriceFM(),
            name="baseline_model",
            train_size=0.7
    ):
        self.df = data_loader.load_data()  
        self.target_columns = target_columns

        train_size = int(len(self.df) * train_size)
        self.train_set, self.test_set = self.df.iloc[:train_size], self.df.iloc[train_size:]

        self.forecast_model = forecast_model
        self.name = name

    def train_model(self) -> None:
        """
        Fits the model with dataframe.
        """
        self.forecast_model.fit(self.train_set)

    def evaluate(self,
                 metric_funcs={
                     'MAE': mean_absolute_error,
                     'MSE': mean_squared_error,
                     'MAPE': mean_absolute_percentage_error
                     }):
        """
        Evaluates model on the test set.
        """
        forecast_values = self.forecast_model.predict(self.test_set.index).to_numpy().reshape(-1)
        true_values = self.test_set[self.target_columns].to_numpy()
        results = dict(keys=metric_funcs.keys())
        for metric, func in metric_funcs.items():
            results[metric] = func(true_values, forecast_values)
        return results

    def plot_forecast(self):
        """
        Plots forecasted data on the test interval as well as the true values.
        """
        forecast_values = self.forecast_model.predict(self.test_set.index).to_numpy().reshape(-1)
        true_values = self.test_set[self.target_columns].to_numpy()

        fig, ax = plt.subplots(nrows=true_values.shape[1], figsize=(12, 8), squeeze=False)

        for target_idx in range(true_values.shape[1]):
            ax[target_idx, 0].plot(self.test_set.index, true_values)
            ax[target_idx, 0].plot(self.test_set.index, forecast_values, linestyle='--')
            ax[target_idx, 0].xaxis.set_major_locator(MonthLocator(interval=1))
            ax[target_idx, 0].xaxis.set_major_formatter(DateFormatter('%b-%Y'))
            ax[target_idx, 0].set_ylabel(self.target_columns[target_idx])
        return fig

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
        target_columns=['Close'],
        forecast_model: ForecastModel = ClosePriceFM(),
        name="baseline_model",
        train_size=0.7
    ):
        """
        (Constructor)
        Loads the model from the given path with the dataframe.

        Parameters:
            path (str): path to the folder with model files.
            df (pd.DataFrame): dataframe for model fitting.
            target_columns (List[str]): list of columns predicted.
            forecast_model (ForecastModel): model forecasting target.
            name (str): name of the model.
            train_size (float): ratio of train set size.

        Returns:
            ForecastFramework: constructed framework object.
        """
        assert os.path.exists(path)
        framework = ForecastFramework(df, target_columns, forecast_model, name, train_size)
        framework.forecast_model.load(df, path)
        return framework

    def create_forecast(self, value: int = 1, unit: str = 'd') -> pd.Series:
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
