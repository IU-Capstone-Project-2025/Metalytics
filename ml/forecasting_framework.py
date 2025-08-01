import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import HourLocator, MinuteLocator, DateFormatter
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import os
from typing import Dict, Any
from forecasting_models import ForecastModel, ClosePriceFM
from data_loader import GoldDataLoader
from filter import SGFilter
import numpy as np


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
    filter: SGFilter

    def __init__(
            self,
            data_loader: GoldDataLoader = GoldDataLoader(),
            target_columns=['Close'],
            forecast_model=ClosePriceFM(),
            name="baseline_model",
            test_size=0
    ):
        """
        (Constructor)

        Parameters:
            path (str): path to the folder with model files.
            df (pd.DataFrame): dataframe for model fitting.
            target_columns (List[str]): list of columns predicted.
            forecast_model (ForecastModel): model forecasting target.
            name (str): name of the model.
            test_size (int): number of last half-hours dedicated for testing.
        """
        self.df = data_loader.load_data().asfreq('30min').bfill().ffill()
        self.filter = SGFilter()
        self.df = self.filter.filter(self.df)

        self.target_columns = target_columns

        if (test_size == 0):
            self.train_set = self.df
            self.test_set = None
        else:
            self.train_set, self.test_set = self.df.iloc[:-test_size], self.df.iloc[-test_size:]

        self.forecast_model = forecast_model
        self.name = name

    def train_model(self) -> None:
        """
        Fits the model with dataframe.
        """
        self.forecast_model.fit(self.train_set)

    def cross_validate(self,
                       params: Dict[str, Any],
                       K: int = 20,
                       test_size: int = 48*5,
                       metric_funcs={
                           'MAE': mean_absolute_error,
                           'MSE': mean_squared_error,
                           'MAPE': mean_absolute_percentage_error
                       }):
        """
        Evaluates model using cross-validation.
        """

        results = dict(keys=metric_funcs.keys())

        for metric in metric_funcs.keys():
            results[metric] = 0

        tss = TimeSeriesSplit(n_splits=K, test_size=test_size, gap=0)

        idx = 0

        for train_idx, test_idx in tss.split(self.df):
            print(f"{idx + 1}-fold:")
            idx += 1
            train_set = self.df.iloc[train_idx]
            test_set = self.df.iloc[test_idx]

            self.forecast_model.fit(df=train_set, params=params)

            forecast_values = self.forecast_model.predict(test_set.index).to_numpy().reshape(-1)
            true_values = test_set[self.target_columns].to_numpy()

            # plot
            fig, ax = plt.subplots(nrows=1, figsize=(12, 8))
            ax.xaxis.set_major_locator(MinuteLocator(interval=30))
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
            ax.plot(test_set.index, true_values, label='true')
            ax.plot(test_set.index, forecast_values, linestyle='--', label='forecast')

            ax.grid()
            fig.legend()
            fig.savefig(f"cross_val_forecasts/forecast_{idx}.png")

            for metric, func in metric_funcs.items():
                results[metric] += func(true_values, forecast_values)

            print(results)

        # Average results over `K` folds
        for metric in metric_funcs.keys():
            results[metric] /= K

        return results

    def evaluate(self,
                 metric_funcs={
                     'MAE': mean_absolute_error,
                     'MSE': mean_squared_error,
                     'MAPE': mean_absolute_percentage_error
                     }):
        """
        Evaluates model on the test set.
        """
        if (self.test_set is None):
            raise ValueError("No test set were provided")
        forecast_values = self.forecast_model.predict(self.test_set.index).to_numpy().reshape(-1)
        true_values = self.test_set[self.target_columns].to_numpy()
        results = dict(keys=metric_funcs.keys())
        for metric, func in metric_funcs.items():
            results[metric] = func(true_values, forecast_values)
        return results

    def forecast_certainty_measure_(self,
                                    y_true: np.array,
                                    y_pred: np.array,
                                    ):
        pass

    def error_threshold_range(self, threshold: float = 1e-3):
        """
        Returns the range of forecasted data having
        percentage error with test set lower than threshold.

        Parameters:
            threshold (float): value in the range (0, 1) to provide an upper bound
            for acceptable errors.

        Returns:
            pd.DatetimeIndex: the date range satisfying threshold error.
        """
        if (self.test_set is None):
            raise ValueError("No test set were provided")
        forecast_values = self.forecast_model.predict(self.test_set.index).to_numpy().reshape(-1)
        true_values = self.test_set[self.target_columns].to_numpy().reshape(-1)

        errors = np.abs(true_values - forecast_values) / true_values

        errors_indexed = pd.Series(errors.ravel(), index=self.test_set.index)

        last_index = errors_indexed[errors_indexed >= threshold].index[0]

        # Check for increasing error
        acceptable_errors = errors_indexed.loc[:last_index]
        if not (acceptable_errors.iloc[1:].to_numpy() >= acceptable_errors.iloc[:-1].to_numpy()).all():
            print("[!!!] Warning: errors in the acceptable range are not ascending")

        return acceptable_errors.index

    def plot_forecast(self):
        """
        Plots forecasted data on the test interval as well as the true values.
        """
        if (self.test_set is None):
            raise ValueError("No test set were provided")
        forecast_values = self.forecast_model.predict(self.test_set.index).to_numpy().reshape(-1)
        true_values = self.test_set[self.target_columns].to_numpy()

        fig, ax = plt.subplots(nrows=true_values.shape[1], figsize=(12, 8), squeeze=False)

        for target_idx in range(true_values.shape[1]):
            ax[target_idx, 0].plot(self.test_set.index, true_values, label='true')
            ax[target_idx, 0].plot(self.test_set.index, forecast_values, linestyle='--', label='forecast')
            ax[target_idx, 0].xaxis.set_major_locator(HourLocator(interval=8))
            ax[target_idx, 0].xaxis.set_major_formatter(DateFormatter('%m-%d, %H'))
            ax[target_idx, 0].set_ylabel(self.target_columns[target_idx])
            ax[target_idx, 0].grid()
        fig.autofmt_xdate(rotation=45)
        fig.legend()
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
        data_loader: GoldDataLoader = GoldDataLoader(),
        target_columns=['Close'],
        forecast_model: ForecastModel = ClosePriceFM(),
        name="baseline_model",
        test_size=0
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
            test_size (int): number of last half-hours dedicated for testing.

        Returns:
            ForecastFramework: constructed framework object.
        """
        assert os.path.exists(path)
        framework = ForecastFramework(data_loader, target_columns, forecast_model, name, test_size)
        framework.forecast_model.load(framework.train_set, path)
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
        Produces date range from the last available observation + 30 min
        to the date after `value` number of `unit`s.

        Parameters:
            value (int): number of units.
            unit (str): unit of time (e.g. 'h', 'd', 'm')

        Returns:
            pd.DatetimeIndex: date range for forecasting model.
        """
        date_index = self.df.index
        return pd.date_range(
            date_index[-1] + pd.Timedelta(value=30, unit='min'),
            date_index[-1] + pd.Timedelta(value=30, unit='min') + pd.Timedelta(value=value, unit=unit),
            freq='30min'
        )
