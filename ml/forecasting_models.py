import pandas as pd
import numpy as np
import joblib
import ta
from numpy.lib.stride_tricks import sliding_window_view
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Union, Callable
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from xgboost import XGBRegressor

import json
import os
from pathlib import Path
import warnings
import pickle
from arch import arch_model
from tensorflow import keras

class ForecastModel(ABC):
    """
    Class for building dataset, fitting the model, dumping, and loading it.
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame, params: Dict[str, Any]):
        """
        Creates dataset from `df` and fits the model with parameters to the dataset.

        Parameters:
            df (pd.DataFrame): dataframe from which dataset is built.
            params: dictionary of parameter names and values of the model.

        Returns:
            self (ForecastModel)
        """
        pass

    @abstractmethod
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Builds train/test dataset from the dataframe.

        Parameters:
            df (pd.DataFrame): dataframe from which dataset is built.

        Returns:
            pd.DataFrame: prepared dataset.
        """
        pass

    @abstractmethod
    def build_forecast_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Builds dataset from the dataframe to make forecasts.

        Parameters:
            df (pd.DataFrame): dataframe from which dataset is built.

        Returns:
            pd.DataFrame: prepared dataset.
        """
        pass

    @abstractmethod
    def cross_validation(self,
                         df: pd.DataFrame,
                         params: Dict[str, Any],
                         K: int = 20,
                         test_size: int = 48*5*1,
                         metric: Callable = mean_squared_error) -> float:
        """
        K-fold cross validation of a given test size to evaluate model performance.

        Parameters:
            df (pd.DataFrame): dataframe.
            params (params: Dict[str, Any]): parameter names and values.
            K (int): number of folds.
            test_size (int): size of test set (in hours).
            metric (Callable): metric function.

        Returns:
            float: averaged value of metric over folds.
        """
        pass

    @abstractmethod
    def predict(self, date_range: pd.DatetimeIndex) -> pd.Series:
        """
        Creates forecast series of a given date range.

        Parameters:
            date_range (pd.DatetimeIndex): range of dates for index.

        Returns:
            pd.Series: forecasted series.
        """
        pass

    @abstractmethod
    def dump(self, path: str) -> None:
        """
        Saves model to a file.

        Parameters:
            path (str): path to the folder containing file.
        """
        pass

    @abstractmethod
    def load(self, df: pd.DataFrame, path: str) -> None:
        """
        Sets the dataframe and loads the model from a file.

        Parameters:
            df (pd.DataFrame): dataframe from which dataset is built.
            path (str): path to the folder containing file.
        """
        pass


class SLFM(ForecastModel):
    """
    Statistical Lag Forecast Model (ARIMA(1,1,1))

    Attributes:
        feature_name (str): name of the target feature.
        model_ (ARIMA): statistical model.
        model_fit (ARIMAResults): fit model.
    """

    feature_name: str
    model_: ARIMA
    model_fit: ARIMAResults

    def __init__(self, feature_name: str):
        self.feature_name = feature_name
        self.model_ = None
        self.model_fit = None

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        prices_reindexed = df[self.feature_name]
        return prices_reindexed

    def build_forecast_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.build(df)

    def cross_validation(self,
                         df: pd.DataFrame,
                         params: Dict[str, Any],
                         K: int = 20, test_size: int = 48*5*1,
                         metric: Callable = mean_squared_error) -> float:
        return None

    def fit(self, df: pd.DataFrame):
        df_ = self.build(df)
        self.model_ = ARIMA(df_, order=(1, 1, 1))
        self.model_fit = self.model_.fit()
        return self

    def predict(self, date_range: pd.DatetimeIndex) -> pd.Series:
        assert self.model_fit is not None
        y_pred = self.model_fit.predict(start=date_range[0], end=date_range[-1], dynamic=False)
        return y_pred

    def dump(self, path: str) -> None:
        self.model_fit.save(f"{path}/{self.feature_name}_predictor.joblib")

    def load(self, df: pd.DataFrame, path: str) -> None:
        self.model_fit = ARIMAResults.load(f"{path}/{self.feature_name}_predictor.joblib")


def decompose_tabular_data(data: np.array, h: int) -> Tuple[np.array, np.array]:
    """
    Compose features from successive dataset objects using sliding window.

    For example, [y_1, y_2, y_3, y_4, ...] (h=2) would produce

        ([ [*y_1, *y_2, y_3[:, 1:]], [*y_2, *y_3, y_4[:, 1:]], ...], [y_3, y_4, ...]).

    Here *a means unpacking values from a vector.

    Parameters:
        data (np.array): 2D-array of target observations.
        h (int): sliding window size.

    Returns:
        Tuple[np.array, np.array]: tuple of sampled dataset of h features
        and the target value for them.
    """
    X = sliding_window_view(data, window_shape=(h, data.shape[1])).reshape(-1, h * data.shape[1])[:-1, :]
    y = data[h:]
    target, features = y[:, 0], y[:, 1:]
    X = np.hstack([X, features, np.ones(shape=(X.shape[0], 1))])
    return (X, target)


def compose_forecast_frame(data: np.array, features: np.array, lag: int) -> Tuple[np.array, np.array]:
    """
    Compose features from the last observations and features of forecast timeframe.

    For example, [..., y_{n-k}, ..., y_{n-3}, y_{n-2}, y_{n-1}, y_n] (lag=2) would produce

        [*y_{n-1}, *y_{n}, *features, 1].

    Here *a means unpacking values from a vector.

    Parameters:
        data (np.array): 2D-array of target observations.
        features (np.array): 1D-array of current forecast timeframe features.
        lag (int): number of lagged observations to include.

    Returns:
        1D-array of features for the model prediction.
    """
    X = data[-lag:].reshape(-1, lag * data.shape[1])
    X = np.hstack([X, features.reshape(1, -1), np.ones(shape=(X.shape[0], 1))])
    return X


class XGBoostFM(ForecastModel):
    """
    XGBRegressor forecasting model for selected target.

    Attributes:
        model_ (XGBRegressor): regression model.
        df_ (pd.DataFrame): dataset built for training.
        target (str): target column.
        stationary (bool): True if series is stationary (first difference is irrelevant).
        lag (int): number of lagged features.
        last_value_ (float): last observed value of the series (for differenced predictions).
    """

    model_: BaseEstimator
    scaler_: MinMaxScaler
    df_: pd.DataFrame
    target: str
    stationary: bool
    lag: int
    last_value_: float

    def __init__(self, target: str, stationary: bool = True, lag: int = 60):
        self.target = target
        self.stationary = stationary
        self.lag = lag
        self.last_value_ = None

    def preprocess_(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocessing of dataframe before building.

        Parameter:
            df (pd.DataFrame): dataframe.

        Returns:
            pd.DataFrame: preprocessed dataframe.
        """

        df = df.copy()

        Q1, Q3 = df.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))

        df.loc[outliers] = np.nan
        df = df.interpolate(method='time')

        return df

    def build(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        df = pd.DataFrame(self.preprocess_(df[self.target]))

        # Obtain first difference (if relevant)
        if (not self.stationary):
            self.last_value_ = df.iloc[-1][self.target]
            df = df.diff().dropna()

        # Day of Week (0=Monday, 6=Sunday)
        df['day_of_week'] = df.index.dayofweek
        df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=range(7), ordered=True)

        df['year'] = df.index.year
        df['year'] = pd.Categorical(df['year'], categories=range(2023, 2026), ordered=True)

        # Month (1-12)
        month_index = df.index.month

        # Cyclical encoding for months (12-month period)
        df['month_sin'] = np.sin(2 * np.pi * month_index / 12)
        df['month_cos'] = np.cos(2 * np.pi * month_index / 12)

        # Season (1=Winter, 2=Spring, 3=Summer, 4=Fall)
        df['season'] = (df.index.month % 12 + 3) // 3
        df['season'] = pd.Categorical(df['season'], categories=range(1, 5), ordered=True)

        # Weekend flag (1 if Saturday/Sunday, else 0)
        df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)

        # Hour
        hour_index = df.index.hour

        # Cyclical encoding for hours (24h period)
        df['hour_sin'] = np.sin(2 * np.pi * hour_index / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour_index / 24)

        # For cyclical features (day_of_week, season)
        df = pd.get_dummies(df, columns=['day_of_week', 'season'], prefix=['dow', 'season'])

        # Normalization
        for feature in ['year', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos']:
            df[feature] = MinMaxScaler().fit_transform(df[[feature]])

        return df.astype(np.float32)

    def build_forecast_data(self, df: pd.DataFrame):
        """
        Builds forecasting dataset (indexed with forecasting date range) from the dataframe.

        Parameters:
            df (pd.DataFrame): dataframe from which dataset is built.

        Returns:
            pd.DataFrame: prepared dataset.
        """

        df = df.copy()

        df = pd.DataFrame(df[self.target])

        # Day of Week (0=Monday, 6=Sunday)
        df['day_of_week'] = df.index.dayofweek
        df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=range(7), ordered=True)

        df['year'] = df.index.year
        df['year'] = pd.Categorical(df['year'], categories=range(2023, 2026), ordered=True)

        # Month (1-12)
        month_index = df.index.month

        # Cyclical encoding for months (12-month period)
        df['month_sin'] = np.sin(2 * np.pi * month_index / 12)
        df['month_cos'] = np.cos(2 * np.pi * month_index / 12)

        # Season (1=Winter, 2=Spring, 3=Summer, 4=Fall)
        df['season'] = (df.index.month % 12 + 3) // 3
        df['season'] = pd.Categorical(df['season'], categories=range(1, 5), ordered=True)

        # Weekend flag (1 if Saturday/Sunday, else 0)
        df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)

        # Hour
        hour_index = df.index.hour

        # Cyclical encoding for hours (24h period)
        df['hour_sin'] = np.sin(2 * np.pi * hour_index / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour_index / 24)

        # For cyclical features (day_of_week, season)
        df = pd.get_dummies(df, columns=['day_of_week', 'season'], prefix=['dow', 'season'])

        # Normalization
        for feature in ['year', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos']:
            df[feature] = MinMaxScaler().fit_transform(df[[feature]])

        return df.astype(np.float32)

    def cross_validation(self,
                         df: pd.DataFrame,
                         params: Dict[str, Any],
                         K: int = 20,
                         test_size: int = 48*5*1,
                         metric: Callable = mean_squared_error) -> float:
        history_df = self.build(df)

        tss = TimeSeriesSplit(n_splits=K, test_size=test_size, gap=0)

        X_train, y_train = decompose_tabular_data(history_df.to_numpy(), h=self.lag)
        Xy = pd.DataFrame(X_train, y_train)

        score = 0

        for train_idx, val_idx in tss.split(Xy):
            train = Xy.iloc[train_idx]
            val = Xy.iloc[val_idx]

            X_train, y_train = train.to_numpy(), train.index.to_numpy()
            X_val, y_val = val.to_numpy(), val.index.to_numpy()

            regressor = XGBRegressor(
                n_estimators=params['n_estimators'],
                learning_rate=params['learning_rate'],
                max_depth=params['max_depth'],
                min_child_weight=params['min_child_weight'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                gamma=params['gamma'],
                scale_pos_weight=np.sqrt(len(y_train)/y_train.sum()),
                tree_method='hist',
                booster='gbtree',
                objective='reg:squarederror',
                random_state=42
            )

            regressor.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)

            y_pred = regressor.predict(X_val)

            score += metric(y_val, y_pred)

        return score / K

    def fit(
        self,
        df: pd.DataFrame,
        params: Dict[str, Union[float, str]] = {
            "n_estimators": 3200,
            "learning_rate": 0.1,
            "max_depth": 4,
            "min_child_weight": 8.7,
            "subsample": 1,
            "colsample_bytree": 1,
            "gamma": 3
        }
    ):
        self.df_ = self.build(df)

        train_size = int(len(self.df_) * 0.8)
        train_set, test_set = self.df_.iloc[:train_size], self.df_.iloc[train_size:]

        X_train, y_train = decompose_tabular_data(train_set.to_numpy(), h=self.lag)
        X_test, y_test = decompose_tabular_data(test_set.to_numpy(), h=self.lag)

        self.model_ = XGBRegressor(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            min_child_weight=params['min_child_weight'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            gamma=params['gamma'],
            scale_pos_weight=np.sqrt(len(y_train)/y_train.sum()),
            tree_method='hist',
            booster=None,
            objective='reg:squarederror',
            random_state=42
        )

        self.model_.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)

        return self

    def predict(self, date_range: pd.DatetimeIndex) -> pd.Series:

        prediction_df = self.build_forecast_data(pd.DataFrame(index=date_range, columns=[self.target]))

        history_df = self.df_.copy()

        for date in date_range:
            feature_columns = [column for column in prediction_df.columns if column != self.target]
            features = prediction_df.loc[date, feature_columns].copy()

            x_ = compose_forecast_frame(history_df.to_numpy(), features.to_numpy(), self.lag)
            y_ = self.model_.predict(x_)

            features.loc[self.target] = y_
            history_df = pd.concat([history_df, features.to_frame().T])

        y_pred = history_df.loc[date_range[0]:date_range[-1], self.target]

        if (self.stationary):
            return y_pred

        return self.last_value_ + np.cumsum(y_pred)

    def dump(self, path: str) -> None:
        joblib.dump(self.model_, f"{path}/{self.target}_predictor.joblib")

    def load(self, df: pd.DataFrame, path: str):
        self.df_ = self.build(df)
        self.model_ = joblib.load(f"{path}/{self.target}_predictor.joblib")


class ClosePriceFM(ForecastModel):
    """
    Forecasting model for `Close` target.

    Selected model: XGBRegressor.

    Attributes:
        model_ (BaseEstimator): regression model.
        feature_models (Dict[str, ForecastModel]): dictionary of auxiliary models and their names.
        df_ (pd.DataFrame): dataset built for training.
        last_close_price (float): price of the last observed close prices.
        lag (int): number of lagged features.
        indicators (List[str]): names of indicators.
    """

    model_: BaseEstimator
    feature_models: Dict[str, ForecastModel]
    df_: pd.DataFrame
    last_close_price: float
    lag: int
    train_size: float
    indicators: List[str] = ['EMA20', 'RSI14', 'ATR14', 'MACD', 'MACD_Signal', 'MACD_Hist']

    def __init__(
            self,
            lag: int = 50,
            train_size: float = 0.7,
            feature_models: Dict[str, ForecastModel] = {
                'High': XGBoostFM('High', stationary=False),
                'Low': XGBoostFM('Low', stationary=False),
                'Volume': XGBoostFM('Volume')
            }
    ):
        self.lag = lag
        self.train_size = train_size
        self.feature_models = feature_models

    def build(self, df: pd.DataFrame):

        df = df.copy()

        # First difference to remove trend
        self.last_close_price = df['Close'].iloc[-1]
        df.loc[:, 'Close'] = df['Close'].diff()
        df = df.dropna()

        return df

    def build_forecast_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Builds forecasting dataset (indexed with forecasting date range) from the dataframe.

        Parameters:
            df (pd.DataFrame): dataframe from which dataset is built.

        Returns:
            pd.DataFrame: prepared dataset.
        """

        df = df.copy()

        # Predict feature targets
        for feature, feature_model in self.feature_models.items():
            df.loc[:, feature] = feature_model.predict(df.index)
        # Set `Open` price
        df.loc[:, 'Open'] = np.nan
        df.loc[df.index[0], 'Open'] = self.df_['Close'].iloc[-1]

        # Set `Close` price
        df.loc[:, 'Close'] = np.nan

        # Set Indicators
        for indicator in self.indicators:
            df.loc[:, indicator] = np.nan
            df.loc[df.index[0], indicator] = self.df_[indicator].iloc[-1]

        return df

    def cross_validation(self,
                         df: pd.DataFrame,
                         params: Dict[str, Any],
                         K: int = 20,
                         test_size: int = 48*5*1,
                         metric: Callable = mean_squared_error) -> float:
        history_df = self.build(df)

        tss = TimeSeriesSplit(n_splits=K, test_size=test_size, gap=0)

        X_train, y_train = decompose_tabular_data(history_df.to_numpy(), h=self.lag)
        Xy = pd.DataFrame(X_train, y_train)

        score = 0

        for train_idx, val_idx in tss.split(Xy):
            train = Xy.iloc[train_idx]
            val = Xy.iloc[val_idx]

            X_train, y_train = train.to_numpy(), train.index.to_numpy()
            X_val, y_val = val.to_numpy(), val.index.to_numpy()

            regressor = XGBRegressor(
                n_estimators=params['n_estimators'],
                learning_rate=params['learning_rate'],
                max_depth=params['max_depth'],
                min_child_weight=params['min_child_weight'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                gamma=params['gamma'],
                tree_method='hist',
                booster=None,
                objective='reg:squarederror',
                random_state=42
            )

            regressor.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)

            y_pred = regressor.predict(X_val)

            score += metric(y_val, y_pred)

        return score / K

    def fit(
        self,
        df: pd.DataFrame,
        params: Dict[str, Union[float, str]] = {
            "n_estimators": 800,
            "learning_rate": 0.1,
            "max_depth": 7,
            "min_child_weight": 4,
            "subsample": 0.7,
            "colsample_bytree": 1.0,
            "gamma": 2
        }
    ):
        self.df_ = self.build(df)

        # Fit feature models
        for feature_model in self.feature_models.values():
            feature_model.fit(df)

        train_size = int(len(self.df_) * self.train_size)
        train_set, test_set = self.df_.iloc[:train_size], self.df_.iloc[train_size:]

        X_train, y_train = decompose_tabular_data(train_set.to_numpy(), h=self.lag)
        X_test, y_test = decompose_tabular_data(test_set.to_numpy(), h=self.lag)

        self.model_ = XGBRegressor(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            min_child_weight=params['min_child_weight'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            gamma=params['gamma'],
            scale_pos_weight=np.sqrt(len(y_train)/y_train.sum()),
            tree_method='hist',
            booster=None,
            objective='reg:squarederror',
            random_state=42,
        )

        self.model_.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)

        return self

    def predict(self, date_range: pd.DatetimeIndex) -> pd.Series:

        prediction_df = self.build_forecast_data(pd.DataFrame(index=date_range))

        history_df = self.df_.copy()

        for idx, date in enumerate(date_range):

            feature_columns = [column for column in prediction_df.columns if column != 'Close']
            features = prediction_df.loc[date, feature_columns].copy()

            x_ = compose_forecast_frame(history_df.to_numpy(), features.to_numpy(), self.lag)
            y_ = self.model_.predict(x_)

            history_close_price = pd.concat([history_df['Close'], pd.Series(y_, index=[date])])
            history_high_price = pd.concat([history_df['High'], pd.Series(features['High'], index=[date])])
            history_low_price = pd.concat([history_df['Low'], pd.Series(features['Low'], index=[date])])

            # Indicators
            features.loc['EMA20'] = ta.trend.EMAIndicator(history_close_price, window=20).ema_indicator().iloc[-1]
            features.loc['RSI14'] = ta.momentum.RSIIndicator(history_close_price, window=14).rsi().iloc[-1]
            features.loc['ATR14'] = ta.volatility.AverageTrueRange(
                history_high_price, history_low_price, history_close_price, window=14
            ).average_true_range().iloc[-1]

            macd = ta.trend.MACD(history_close_price, window_slow=26, window_fast=12, window_sign=9)
            features.loc['MACD'] = macd.macd().iloc[-1]
            features.loc['MACD_Signal'] = macd.macd_signal().iloc[-1]
            features.loc['MACD_Hist'] = macd.macd_diff().iloc[-1]

            # Set future indicator values as current
            if idx < len(date_range)-1:
                for indicator in self.indicators:
                    prediction_df.loc[date_range[idx+1], indicator] = features[indicator]
                # Set `Open` price
                prediction_df.loc[date_range[idx+1], 'Open'] = history_df['Close'].iloc[-1]

            features.loc['Close'] = y_
            history_df = pd.concat([history_df, pd.DataFrame.from_records([features], index=[date])])

        # Recover original time series
        y_pred = history_df.loc[date_range[0]:date_range[-1], 'Close']
        price_prediction = self.last_close_price + np.cumsum(y_pred)

        return price_prediction

    def dump(self, path: str) -> None:
        joblib.dump(self.model_, f"{path}/Close_predictor.joblib")
        for model in self.feature_models.values():
            model.dump(path)

    def load(self, df: pd.DataFrame, path: str):
        self.df_ = self.build(df.copy())
        self.model_ = joblib.load(f"{path}/Close_predictor.joblib")
        for model in self.feature_models.values():
            model.load(df, path)


class LSTMCloseFM(ForecastModel):
    def __init__(self, train_size=0.7):
        self.train_size = train_size
        self.df_: pd.DataFrame = None
        self.last_close_price: float = None
        self.last_open_price: float = None
        self.last_high_price: float = None
        self.last_low_price: float = None
        self.target_names = [
            'Close', 'High', 'Low', 'Volume'
        ]
        self.indicators = [
            'EMA20', 'RSI14', 'ATR14', 'MACD', 'MACD_Signal', 'MACD_Hist'
        ]
        self.feature_names = self.target_names + self.indicators + [
            'year', 'month_sin', 'month_cos', 'hour_sin',
            'hour_cos', 'dow_sin', 'dow_cos', 'season_sin', 'season_cos'
        ]
        self.epochs = 50
        self.patience = 10

    def _build_model(self, params: Dict[str, Any]):
        model = Sequential([
            Input(shape=(len(self.feature_names), params['lag'])),
            LSTM(params['layer1'], return_sequences=True),
            Dropout(params['dropout']),
            LSTM(params['layer2']),
            Dropout(params['dropout']),
            Dense(params['layer3'], activation='relu'),
            Dense(len(self.target_names))
        ])
        model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mse')
        return model

    def build(self, df: pd.DataFrame):

        df = df.copy()

        # First difference to remove trend
        self.last_close_price = df['Close'].iloc[-1]
        self.last_open_price = df['Open'].iloc[-1]
        self.last_high_price = df['High'].iloc[-1]
        self.last_low_price = df['Low'].iloc[-1]

        # First difference for target variables
        for col in self.target_names:
            df[col] = df[col].diff()
        df = df.dropna()

        df['year'] = df.index.year
        df['year'] = pd.Categorical(df['year'], categories=range(2023, 2026), ordered=True)

        # Month (1-12)
        month_index = df.index.month

        # Cyclical encoding for months (12-month period)
        df['month_sin'] = np.sin(2 * np.pi * month_index / 12)
        df['month_cos'] = np.cos(2 * np.pi * month_index / 12)

        # Season (1=Winter, 2=Spring, 3=Summer, 4=Fall)
        season_index = (df.index.month % 12 + 3) // 3

        # Season (1-4) → cyclical
        df['season_sin'] = np.sin(2 * np.pi * season_index / 4)
        df['season_cos'] = np.cos(2 * np.pi * season_index / 4)

        # Day of week (0-6) → cyclical
        df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

        # Hour
        hour_index = df.index.hour

        # Cyclical encoding for hours (24h period)
        df['hour_sin'] = np.sin(2 * np.pi * hour_index / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour_index / 24)

        df = df.astype(np.float32)

        return df

    def build_forecast_data(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        df['year'] = df.index.year
        df['year'] = pd.Categorical(df['year'], categories=range(2023, 2026), ordered=True)

        # Month (1-12)
        month_index = df.index.month

        # Cyclical encoding for months (12-month period)
        df['month_sin'] = np.sin(2 * np.pi * month_index / 12)
        df['month_cos'] = np.cos(2 * np.pi * month_index / 12)

        # Season (1=Winter, 2=Spring, 3=Summer, 4=Fall)
        season_index = (df.index.month % 12 + 3) // 3

        # Season (1-4) → cyclical
        df['season_sin'] = np.sin(2 * np.pi * season_index / 4)
        df['season_cos'] = np.cos(2 * np.pi * season_index / 4)

        # Day of week (0-6) → cyclical
        df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

        # Hour
        hour_index = df.index.hour

        # Cyclical encoding for hours (24h period)
        df['hour_sin'] = np.sin(2 * np.pi * hour_index / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour_index / 24)

        # Set `Open` price
        df.loc[:, 'Open'] = np.nan
        df.loc[df.index[0], 'Open'] = self.df_['Close'].iloc[-1]

        # Set Indicators
        for indicator in self.indicators:
            df.loc[:, indicator] = np.nan
            df.loc[df.index[0], indicator] = self.df_[indicator].iloc[-1]

        df = df.astype(np.float32)

        return df

    def _prepare_data(self, df: pd.DataFrame, lag: int):
        """
        Split dataset into a tensor of lagged observations.
        """

        data = df.copy()

        target_data = data[self.target_names].copy()

        X = sliding_window_view(data.to_numpy(), window_shape=lag, axis=0)
        X = X[:-1]
        y = target_data[lag:].to_numpy()

        return X, y

    def cross_validation(self,
                         df: pd.DataFrame,
                         params: Dict[str, Any],
                         K: int = 20, test_size: int = 48*5*1,
                         metric: Callable = mean_squared_error) -> float:
        history_df = self.build(df).copy()

        tss = TimeSeriesSplit(n_splits=K, test_size=test_size, gap=0)

        X, y = self._prepare_data(history_df, lag=params['lag'])

        score = 0

        for train_idx, val_idx in tss.split(X):

            X_train, X_val = X[train_idx][:-params['lag']], X[val_idx][:-params['lag']]
            y_train, y_val = y[train_idx][params['lag']:], y[val_idx][params['lag']:]

            self.model = self._build_model(params)

            self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=64,
                callbacks=[EarlyStopping(patience=self.patience, restore_best_weights=True)],
                verbose=0
            )

            y_pred = self.model.predict(X_val, verbose=0)

            score += metric(y_val, y_pred)

        return score / K

    def fit(self,
            df: pd.DataFrame,
            params: Dict[str, Any] = {
                'layer1': 128,
                'layer2': 64,
                'layer3': 32,
                'dropout': 0.3,
                'learning_rate': 0.01,
                'lag': 60
            },
            ):

        self.df_ = self.build(df)
        train_size = int(len(self.df_) * self.train_size)
        train_set, val_set = self.df_.iloc[:train_size], self.df_.iloc[train_size:]

        X_train, y_train = self._prepare_data(train_set, lag=params['lag'])
        X_val, y_val = self._prepare_data(val_set, lag=params['lag'])

        self.model = self._build_model(params)
        self.params = params

        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=64,
            callbacks=[EarlyStopping(patience=self.patience, restore_best_weights=True)],
            verbose=1
        )
        return self

    def predict(self, date_range: pd.DatetimeIndex) -> pd.Series:

        prediction_df = self.build_forecast_data(pd.DataFrame(index=date_range))

        history_df = self.df_.copy()

        for idx, date in enumerate(date_range):
            x_ = history_df.iloc[-self.params['lag']:].to_numpy().reshape(1, -1, self.params['lag'])
            y_ = self.model.predict(x_, verbose=0)[0]

            prediction_df.loc[date, self.target_names] = y_

            history_df = pd.concat([history_df, pd.DataFrame.from_records([prediction_df.loc[date]], index=[date])])

            # Indicators
            history_df['EMA20'] = ta.trend.EMAIndicator(history_df['Close'], window=20).ema_indicator()
            history_df['RSI14'] = ta.momentum.RSIIndicator(history_df['Close'], window=14).rsi()
            history_df['ATR14'] = ta.volatility.AverageTrueRange(
                history_df['High'], history_df['Low'], history_df['Close'], window=14
            ).average_true_range()

            macd = ta.trend.MACD(history_df['Close'], window_slow=26, window_fast=12, window_sign=9)
            history_df['MACD'] = macd.macd()
            history_df['MACD_Signal'] = macd.macd_signal()
            history_df['MACD_Hist'] = macd.macd_diff()

            # Set `Open` price
            if idx < len(date_range)-1:
                prediction_df.loc[date_range[idx+1], 'Open'] = history_df['Close'].iloc[-1]

        # Recover original time series
        y_pred = history_df.loc[date_range[0]:date_range[-1]]
        close_price_cumulative = np.cumsum(y_pred)
        close_price_prediction = self.last_close_price + close_price_cumulative['Close']

        # Other targets
        # open_price_prediction = self.last_close_price + close_price_cumulative['Open']
        # high_price_prediction = self.last_close_price + close_price_cumulative['High']
        # low_price_prediction = self.last_close_price + close_price_cumulative['Low']

        return close_price_prediction

    def dump(self, path: str) -> None:
        self.model.save(f"{path}/LSTM_Close_model.keras")
        # joblib.dump(self.scaler, f"{path}/LSTM_Close_scaler.pkl")
        with open(f"{path}/params.config", "w") as f:
            f.write(json.dumps(self.params, indent=4))

    def load(self, df: pd.DataFrame, path: str):
        self.df_ = self.build(df.copy())
        self.model = load_model(f"{path}/LSTM_Close_model.keras")
        # self.scaler = joblib.load(f"{path}/LSTM_Close_scaler.pkl")
        with open(f"{path}/params.config", "r") as f:
            self.params = json.loads(f.read())


class ClosePriceFM_Silver(ForecastModel):
    """
    Forecasting model for `Close` target.

    Selected model: XGBRegressor.

    Attributes:
        model_ (BaseEstimator): regression model.
        feature_models (Dict[str, ForecastModel]): dictionary of auxiliary models and their names.
        df_ (pd.DataFrame): dataset built for training.
        last_close_price (float): price of the last observed close prices.
        lag (int): number of lagged features.
        indicators (List[str]): names of indicators.
    """

    model_: BaseEstimator
    feature_models: Dict[str, ForecastModel]
    df_: pd.DataFrame
    last_close_price: float
    lag: int
    train_size: float
    indicators: List[str] = ['EMA20', 'RSI14', 'ATR14', 'MACD', 'MACD_Signal', 'MACD_Hist']

    def __init__(
            self,
            lag: int = 50,
            train_size: float = 1.0,
            feature_models: Dict[str, ForecastModel] = {
                'High': XGBoostFM('High', stationary=False),
                'Low': XGBoostFM('Low', stationary=False),
                'Volume': XGBoostFM('Volume')
            }
    ):
        self.lag = lag
        self.train_size = train_size
        self.feature_models = feature_models

    def build(self, df: pd.DataFrame):

        df = df.copy()
        if 'SP500' in df.columns:
            df = df.drop(columns=['SP500'])
        # First difference to remove trend
        self.last_close_price = df['Close'].iloc[-1]
        df.loc[:, 'Close'] = df['Close'].diff()
        
        #if 'SP500' in df.columns:
        #    df['SP500'] = pd.to_numeric(df['SP500'], errors='coerce').interpolate(method='time')
            
        df = df.dropna()

        return df

    def build_forecast_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Builds forecasting dataset (indexed with forecasting date range) from the dataframe.

        Parameters:
            df (pd.DataFrame): dataframe from which dataset is built.

        Returns:
            pd.DataFrame: prepared dataset.
        """

        df = df.copy()
        if 'SP500' in df.columns:
            df = df.drop(columns=['SP500'])
        # Predict feature targets
        for feature, feature_model in self.feature_models.items():
            df.loc[:, feature] = feature_model.predict(df.index)
        # Set `Open` price
        df.loc[:, 'Open'] = np.nan
        df.loc[df.index[0], 'Open'] = self.df_['Close'].iloc[-1]

        # Set `Close` price
        df.loc[:, 'Close'] = np.nan

        # Set Indicators
        for indicator in self.indicators:
            df.loc[:, indicator] = np.nan
            #if indicator == 'SP500':
                # Use last known SP500 value or predict it if needed
               # df.loc[df.index[0], 'SP500'] = self.df_['SP500'].iloc[-1] if 'SP500' in self.df_.columns else np.nan
            #else:
                #df.loc[df.index[0], indicator] = self.df_[indicator].iloc[-1]
            df.loc[df.index[0], indicator] = self.df_[indicator].iloc[-1]

        return df

    def cross_validation(self,
                         df: pd.DataFrame,
                         params: Dict[str, Any],
                         K: int = 20,
                         test_size: int = 48*5*1,
                         metric: Callable = mean_squared_error) -> float:
        history_df = self.build(df)

        tss = TimeSeriesSplit(n_splits=K, test_size=test_size, gap=0)

        X_train, y_train = decompose_tabular_data(history_df.to_numpy(), h=self.lag)
        Xy = pd.DataFrame(X_train, y_train)

        score = 0

        for train_idx, val_idx in tss.split(Xy):
            train = Xy.iloc[train_idx]
            val = Xy.iloc[val_idx]

            X_train, y_train = train.to_numpy(), train.index.to_numpy()
            X_val, y_val = val.to_numpy(), val.index.to_numpy()

            regressor = XGBRegressor(
                n_estimators=params['n_estimators'],
                learning_rate=params['learning_rate'],
                max_depth=params['max_depth'],
                min_child_weight=params['min_child_weight'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                gamma=params['gamma'],
                tree_method='hist',
                booster=None,
                objective='reg:squarederror',
                random_state=42
            )

            regressor.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)

            y_pred = regressor.predict(X_val)

            score += metric(y_val, y_pred)

        return score / K

    def fit(
        self,
        df: pd.DataFrame,
        params: Dict[str, Union[float, str]] = {
            "n_estimators": 800,
            "learning_rate": 0.1,
            "max_depth": 7,
            "min_child_weight": 4,
            "subsample": 0.7,
            "colsample_bytree": 1.0,
            "gamma": 2
        }
    ):
        self.df_ = self.build(df)

        # Fit feature models
        for feature_model in self.feature_models.values():
            feature_model.fit(df)

        train_size = int(len(self.df_) * self.train_size)
        train_set, test_set = self.df_.iloc[:train_size], self.df_.iloc[train_size:]

        X_train, y_train = decompose_tabular_data(train_set.to_numpy(), h=self.lag)
        X_test, y_test = decompose_tabular_data(test_set.to_numpy(), h=self.lag)

        self.model_ = XGBRegressor(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            min_child_weight=params['min_child_weight'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            gamma=params['gamma'],
            scale_pos_weight=np.sqrt(len(y_train)/y_train.sum()),
            tree_method='hist',
            booster=None,
            objective='reg:squarederror',
            random_state=42,
        )

        self.model_.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)

        return self

    def predict(self, date_range: pd.DatetimeIndex, initial_price: float = None) -> pd.Series:
        # Убедимся, что date_range валиден и не содержит старых дат
        date_range = date_range[date_range >= self.df_.index[-1]]
        if len(date_range) == 0:
            raise ValueError("date_range contains only past dates")
        
        prediction_df = self.build_forecast_data(pd.DataFrame(index=date_range))
        history_df = self.df_.copy()
        y_pred = []
        initial_price = initial_price if initial_price is not None else self.last_close_price
        max_volatility = initial_price * 0.02  # 3% от начальной цены за 48 часов
        max_step_change = initial_price * 0.0003  # 0.1% за шаг 
        min_price = initial_price * 0.93  # Минимальная цена 
        
        for idx, date in enumerate(date_range):
            feature_columns = [column for column in prediction_df.columns if column != 'Close']
            features = prediction_df.loc[date, feature_columns].copy()
            x_ = compose_forecast_frame(history_df.to_numpy(), features.to_numpy(), self.lag)
            y_ = self.model_.predict(x_)[0]
            if idx > 0:
                alpha = 0.05  # Коэффициент сглаживания (0 < alpha < 1, меньшие значения - более плавное сглаживание)
                y_ = alpha * y_ + (1 - alpha) * y_pred[-1]
            # Умеренные колебания
            atr = features['ATR14'] if 'ATR14' in features else 0.05
            noise = np.random.normal(0, atr * 0.02)  # Очень слабый шум
            sin_factor = np.sin(idx / 8) * max_volatility * 0.05  # Период ~16 часов
            y_ = y_ + noise + sin_factor
            
            # Ограничиваем изменение за шаг
            y_ = np.clip(y_, -max_step_change, max_step_change)
            
            # Ограничиваем волатильность в окне 48 часов (96 интервалов)
            current_price = initial_price + np.sum(y_pred) + y_ if y_pred else initial_price + y_
            if len(y_pred) >= 96:
                window = y_pred[-95:] + [y_]
                window_price = initial_price + np.sum(window)
                window_min = initial_price - max_volatility
                window_max = initial_price + max_volatility
                if window_price > window_max:
                    y_ = window_max - (initial_price + np.sum(y_pred) if y_pred else initial_price)
                elif window_price < window_min:
                    y_ = window_min - (initial_price + np.sum(y_pred) if y_pred else initial_price)
            
            # Гарантируем положительные цены
            if current_price < min_price:
                y_ = min_price - (initial_price + np.sum(y_pred) if y_pred else initial_price)
            
            y_pred.append(y_)
            
            # Обновляем историю и индикаторы
            history_close_price = pd.concat([history_df['Close'], pd.Series(y_, index=[date])])
            history_high_price = pd.concat([history_df['High'], pd.Series(features['High'], index=[date])])
            history_low_price = pd.concat([history_df['Low'], pd.Series(features['Low'], index=[date])])
            features.loc['EMA20'] = ta.trend.EMAIndicator(history_close_price, window=20).ema_indicator().iloc[-1]
            features.loc['RSI14'] = ta.momentum.RSIIndicator(history_close_price, window=14).rsi().iloc[-1]
            features.loc['ATR14'] = ta.volatility.AverageTrueRange(
                history_high_price, history_low_price, history_close_price, window=14
            ).average_true_range().iloc[-1]
            macd = ta.trend.MACD(history_close_price, window_slow=26, window_fast=12, window_sign=9)
            features.loc['MACD'] = macd.macd().iloc[-1]
            features.loc['MACD_Signal'] = macd.macd_signal().iloc[-1]
            features.loc['MACD_Hist'] = macd.macd_diff().iloc[-1]
            if idx < len(date_range)-1:
                for indicator in self.indicators:
                    prediction_df.loc[date_range[idx+1], indicator] = features[indicator]
                prediction_df.loc[date_range[idx+1], 'Open'] = history_close_price.iloc[-1]
            features.loc['Close'] = y_
            new_row = pd.DataFrame.from_records([features], index=[date])
            history_df = pd.concat([history_df, new_row])
            history_df = history_df.loc[~history_df.index.duplicated(keep='last')]
        
        y_pred = pd.Series(y_pred, index=date_range)
        smoothed_y_pred = y_pred.rolling(window=5, center=True, min_periods=1).mean()  # Скользящее среднее с окном 5
        
        price_prediction = initial_price + np.cumsum(smoothed_y_pred)
        return price_prediction

    def dump(self, path: str) -> None:
        joblib.dump(self.model_, f"{path}/Close_predictor.joblib")
        for model in self.feature_models.values():
            model.dump(path)

    def load(self, df: pd.DataFrame, path: str):
        self.df_ = self.build(df.copy())
        self.model_ = joblib.load(f"{path}/Close_predictor.joblib")
        for model in self.feature_models.values():
            model.load(df, path)


class ZincForecastModel(ForecastModel):
    """
    Конкретная реализация для прогноза цены цинка.
    Совмещает GARCH(1,1) + MinMax‑масштабы + 2‑слойный LSTM.
    """

    # ---------- инициализация ------------------------------------------------
    def __init__(self, window_size: int = 48, split_date: str | pd.Timestamp = "2024-07-01"):
        self.window_size = window_size
        self.split_date = pd.to_datetime(split_date)
        # будет заполнено в build / fit
        self.data_df: pd.DataFrame | None = None
        self.train_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None

        self.feature_cols: List[str] = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Zn_Price",
            "DXY",
            "Brent_USD",
            "Zinc_Production",
            "LogReturn",
            "GARCH_vol",
        ]
        self.scalers: Dict[str, MinMaxScaler] = {c: MinMaxScaler() for c in self.feature_cols}
        self.garch_res = None  # type: ignore
        self.lstm: keras.Model | None = None

    # ---------- приватные хелперы ------------------------------------------
    def _scale_scalar(self, col: str, value: float) -> float:
        """Безопасно масштабирует скаляр (предотвращает предупреждения sklearn)."""
        arr = self.scalers[col].transform(pd.DataFrame([[value]], columns=[col]))
        return float(arr[0, 0])

    def _make_sequences(self, df_scaled: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Формирует (X, y) для LSTM из уже масштабированного df."""
        arr = df_scaled[self.feature_cols].values
        X, y = [], []
        for i in range(self.window_size, len(arr)):
            X.append(arr[i - self.window_size : i])
            y.append(arr[i, self.feature_cols.index("Close")])
        return np.asarray(X, np.float32), np.asarray(y, np.float32)

    # ---------- build -------------------------------------------------------
    def build(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        *Добавляет* LogReturn и GARCH‑волатильность,
        делит на train/test, масштабирует все признаки.
        Возвращает (train_scaled, test_scaled).
        """
        # 1) Копируем, чтобы не портить оригинал
        df = df.copy()
        df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
        df["LogReturn"].fillna(0.0, inplace=True)

        # 2) Fit/forecast GARCH (только один раз на train)
        train_mask = df["DateTime"] < self.split_date
        returns_train = df.loc[train_mask, "LogReturn"].values
        self.garch_res = arch_model(
            returns_train, p=1, q=1, mean="Zero", vol="GARCH", dist="normal"
        ).fit(disp="off")

        sigma_train = self.garch_res.conditional_volatility
        sigma_test = np.sqrt(
            self.garch_res.forecast(horizon=(~train_mask).sum(), reindex=False)
            .variance.values[-1]
        )
        df["GARCH_vol"] = np.concatenate([sigma_train, sigma_test])

        # 3) Разделяем на train/test
        self.data_df = df
        self.train_df = df.loc[train_mask].copy()
        self.test_df = df.loc[~train_mask].copy()

        # 4) Масштабирование
        train_scaled = self.train_df.copy()
        test_scaled = self.test_df.copy()
        for col in self.feature_cols:
            train_scaled[col] = self.scalers[col].fit_transform(self.train_df[[col]])
            test_vals = self.scalers[col].transform(self.test_df[[col]])
            test_scaled[col] = np.clip(test_vals, 0.0, 1.0)
        return train_scaled, test_scaled

    # ---------- build_forecast_data ----------------------------------------
    def build_forecast_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Быстро готовит новый df для roll‑out‑прогноза,
        используя уже обученные скейлеры и GARCH‑модель.
        """
        if self.garch_res is None:
            raise RuntimeError("Модель ещё не обучена: сначала вызовите .fit() или .load()")

        df = df.copy()
        df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
        df["LogReturn"].fillna(0.0, inplace=True)

        # Прогнозируем σ на всю длину df
        horizon = len(df)
        sigma = np.sqrt(
            self.garch_res.forecast(horizon=horizon, reindex=False).variance.values[-1]
        )
        df["GARCH_vol"] = sigma

        # Масштабируем
        for col in self.feature_cols:
            df[col] = self.scalers[col].transform(df[[col]])
        return df

    # ---------- fit ---------------------------------------------------------
    def fit(self, df: pd.DataFrame, params: Dict[str, Any]):
        """
        Полный цикл обучения: build → LSTM fit.
        `params` может содержать:
            - lstm_units: int | tuple[int,int]
            - dropout: float
            - epochs: int
            - batch_size: int
            - patience: int (EarlyStopping)
        """
        train_scaled, _ = self.build(df)  # готовим данные

        # Извлекаем гиперпараметры с дефолтами
        units = params.get("lstm_units", (50, 50))
        if isinstance(units, int):
            units = (units, units)
        dropout = params.get("dropout", 0.2)
        epochs = params.get("epochs", 50)
        batch_size = params.get("batch_size", 64)
        patience = params.get("patience", 5)

        # Подготовка X, y
        X_train, y_train = self._make_sequences(train_scaled)

        # Построение модели
        self.lstm = keras.Sequential(
            [
                keras.layers.Input(shape=(self.window_size, len(self.feature_cols))),
                keras.layers.LSTM(units[0], return_sequences=True),
                keras.layers.Dropout(dropout),
                keras.layers.LSTM(units[1]),
                keras.layers.Dropout(dropout),
                keras.layers.Dense(1),
            ]
        )
        self.lstm.compile(optimizer="adam", loss="mse")
        es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        self.lstm.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[es],
            shuffle=True,
            verbose=2,
        )
        return self

    # ---------- cross_validation -------------------------------------------
    def cross_validation(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        K: int = 20,
        test_size: int = 24 * 10 * 1,
        metric: Callable = mean_squared_error,
    ) -> float:
        """
        Скользящая валидация: каждые `test_size` часов берём новый fold.
        GARCH обучается **один** раз (быстрее), LSTM переобучается на каждом фолде.
        """
        # фиксируем build + GARCH + скейлеры один раз
        self.build(df)
        scores: List[float] = []

        total_len = len(self.train_df)  # type: ignore
        step = test_size // 2  # половинный сдвиг

        for start in range(0, total_len - test_size - self.window_size, step):
            end_train = start + test_size
            train_part = self.train_df.iloc[:end_train]  # type: ignore

            # масштабируем part теми же скейлерами
            train_scaled = train_part.copy()
            for col in self.feature_cols:
                train_scaled[col] = self.scalers[col].transform(train_part[[col]])

            X_tr, y_tr = self._make_sequences(train_scaled)

            # val set = следующие `test_size` часов
            val_part = self.train_df.iloc[end_train - self.window_size : end_train + test_size]  # type: ignore
            val_scaled = val_part.copy()
            for col in self.feature_cols:
                val_scaled[col] = self.scalers[col].transform(val_part[[col]])
            X_val, y_val = self._make_sequences(val_scaled)

            # новая LSTM (маленькая, чтобы было быстрее)
            tmp_model = keras.Sequential(
                [
                    keras.layers.Input(shape=(self.window_size, len(self.feature_cols))),
                    keras.layers.LSTM(30, return_sequences=True),
                    keras.layers.LSTM(30),
                    keras.layers.Dense(1),
                ]
            )
            tmp_model.compile(optimizer="adam", loss="mse")
            tmp_model.fit(
                X_tr,
                y_tr,
                epochs=params.get("epochs", 15),
                batch_size=params.get("batch_size", 64),
                verbose=0,
            )
            y_pred = tmp_model.predict(X_val, verbose=0)
            y_true = y_val

            # обратное масштабирование Close
            y_pred_un = self.scalers["Close"].inverse_transform(y_pred)
            y_true_un = self.scalers["Close"].inverse_transform(y_true.reshape(-1, 1))
            scores.append(metric(y_true_un, y_pred_un))

        return float(np.mean(scores))

    # ---------- predict -----------------------------------------------------
    def predict(self, date_range: pd.DatetimeIndex) -> pd.Series:
        if self.lstm is None or self.data_df is None:
            raise RuntimeError("Модель не обучена или не загружена")

        # 1) стартовое окно: последние window_size часов из history
        hist = self.data_df[self.feature_cols].iloc[-self.window_size :].copy()
        hist_scaled = hist.copy()
        for col in self.feature_cols:
            hist_scaled[col] = self.scalers[col].transform(hist[[col]])
        seq = hist_scaled.values

        last_close = hist["Close"].iloc[-1]
        last_vol = hist["GARCH_vol"].iloc[-1]
        last_macros = hist.iloc[-1][
            ["Zn_Price", "DXY", "Brent_USD", "Zinc_Production", "Volume"]
        ]

        preds = []
        for _ in date_range:
            X_in = seq.reshape(1, self.window_size, len(self.feature_cols))
            close_scaled = self.lstm.predict(X_in, verbose=0)[0, 0].item()
            close_unscaled = self.scalers["Close"].inverse_transform([[close_scaled]])[0, 0]
            preds.append(close_unscaled)

            new_row = {
                "Open": last_close,
                "Close": close_unscaled,
                "High": max(last_close, close_unscaled),
                "Low": min(last_close, close_unscaled),
                "Volume": float(last_macros["Volume"]),
                "Zn_Price": float(last_macros["Zn_Price"]),
                "DXY": float(last_macros["DXY"]),
                "Brent_USD": float(last_macros["Brent_USD"]),
                "Zinc_Production": float(last_macros["Zinc_Production"]),
                "LogReturn": np.log(close_unscaled / last_close) if last_close != 0 else 0.0,
                "GARCH_vol": float(last_vol),
            }
            new_row_scaled = [self._scale_scalar(c, new_row[c]) for c in self.feature_cols]
            seq = np.vstack([seq[1:], new_row_scaled])
            last_close = close_unscaled  # shift

        return pd.Series(preds, index=date_range, name="Pred_Close")

    # ---------- dump / load --------------------------------------------------
    def dump(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        # LSTM
        if self.lstm is None:
            raise RuntimeError("Нет модели для сохранения")
        self.lstm.save(os.path.join(path, "lstm_zinc.keras"), include_optimizer=True, save_format="keras")
        # GARCH + scalers
        with open(os.path.join(path, "garch_and_scalers.pkl"), "wb") as f:
            pickle.dump(
                {
                    "garch_result": self.garch_res,
                    "split_date": self.split_date,
                    "feature_cols": self.feature_cols,
                    "scalers": self.scalers,
                    "window_size": self.window_size,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        # мета‑json
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"window_size": self.window_size, "split_date": str(self.split_date)}, f, indent=2
            )

    def load(self, df: pd.DataFrame, path: str) -> None:
        # восстанавливаем скейлеры + GARCH
        with open(os.path.join(path, "garch_and_scalers.pkl"), "rb") as f:
            saved = pickle.load(f)
        self.garch_res = saved["garch_result"]
        self.scalers = saved["scalers"]
        self.feature_cols = saved["feature_cols"]
        self.window_size = saved["window_size"]
        self.split_date = saved["split_date"]

        # LSTM
        self.lstm = keras.models.load_model(os.path.join(path, "lstm_zinc.keras"))
        # rebuild dataframes so that .predict() работает
        self.build(df)  # пересчитаем LogReturn + GARCH_vol, но НЕ переобучаем