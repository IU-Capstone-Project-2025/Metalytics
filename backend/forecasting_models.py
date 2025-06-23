import pandas as pd
import numpy as np
import joblib
import ta
from numpy.lib.stride_tricks import sliding_window_view
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from xgboost import XGBRegressor


class ForecastModel(ABC):
    """
    Class for building dataset, fitting the model, dumping, and loading it.
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame):
        """
        Creates dataset from `df` and fits the model to the dataset.

        Parameters:
            df (pd.DataFrame): dataframe from which dataset is built.

        Returns:
            self (ForecastModel)
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
        """
        Builds dataset from the dataframe.

        Parameters:
            df (pd.DataFrame): dataframe from which dataset is built.

        Returns:
            pd.DataFrame: prepared dataset.
        """
        prices_reindexed = df[self.feature_name].asfreq('h')
        return prices_reindexed

    def fit(self, df: pd.DataFrame):
        df_ = self.build(df.copy())
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


class VolumeFM(ForecastModel):
    """
    Forecasting model for `Volume` target.

    Model selected: XGBRegressor

    Attributes:
        model_ (BaseEstimator): regression model.
        df_ (pd.DataFrame): dataset built for training
        lag (int): number of lagged features.
    """

    model_: BaseEstimator
    df_: pd.DataFrame
    lag: int

    def __init__(self, lag: int = 25):
        self.lag = lag

    def preprocess_(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocessing of dataframe before building.

        Parameter:
            df (pd.DataFrame): dataframe.

        Returns:
            pd.DataFrame: preprocessed dataframe.
        """
        Q1, Q3 = df.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))

        df[outliers] = np.nan
        df = df.interpolate(method='time')

        return df

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Builds dataset from the dataframe.

        Parameters:
            df (pd.DataFrame): dataframe from which dataset is built.

        Returns:
            pd.DataFrame: prepared dataset.
        """

        df = pd.DataFrame(self.preprocess_(df['Volume']))
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

        # Hour (if your data is intraday)
        hour_index = df.index.hour

        # Cyclical encoding for hours (24h period)
        df['hour_sin'] = np.sin(2 * np.pi * hour_index / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour_index / 24)

        # For cyclical features (day_of_week, month, season)
        df = pd.get_dummies(df, columns=['day_of_week', 'season'], prefix=['dow', 'season'])

        # Scaling is removed to retain original scales in the output
        # df['Volume'] = StandardScaler().fit_transform(df[['Volume']])
        for feature in ['year', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos']:
            df[feature] = MinMaxScaler().fit_transform(df[[feature]])

        return df

    def build_forecast_data(self, df: pd.DataFrame):
        """
        Builds forecasting dataset (indexed with forecasting date range) from the dataframe.

        Parameters:
            df (pd.DataFrame): dataframe from which dataset is built.

        Returns:
            pd.DataFrame: prepared dataset.
        """

        df = pd.DataFrame(df['Volume'])
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

        # Hour (if your data is intraday)
        hour_index = df.index.hour

        # Cyclical encoding for hours (24h period)
        df['hour_sin'] = np.sin(2 * np.pi * hour_index / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour_index / 24)

        # For cyclical features (day_of_week, month, season)
        df = pd.get_dummies(df, columns=['day_of_week', 'season'], prefix=['dow', 'season'])

        # Scaling is removed to retain original scales in the output
        # df['Volume'] = StandardScaler().fit_transform(df[['Volume']])
        for feature in ['year', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos']:
            df[feature] = MinMaxScaler().fit_transform(df[[feature]])

        return df

    def fit(self, df: pd.DataFrame):
        self.df_ = self.build(df.copy())

        X_train, y_train = decompose_tabular_data(self.df_.to_numpy(), h=self.lag)

        train_size = int(len(self.df_) * 0.8)
        train_set, test_set = self.df_.iloc[:train_size], self.df_.iloc[train_size:]

        X_train, y_train = decompose_tabular_data(train_set.to_numpy(), h=self.lag)
        X_test, y_test = decompose_tabular_data(test_set.to_numpy(), h=self.lag)

        self.model_ = XGBRegressor(
            base_score=.5,
            booster='gbtree',
            early_stopping_rounds=10,
            objective='reg:squarederror',
            max_depth=3,
            learning_rate=.05
        )

        self.model_.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)

        return self

    def predict(self, date_range: pd.DatetimeIndex) -> pd.Series:

        prediction_df = self.build_forecast_data(pd.DataFrame(index=date_range, columns=['Volume']))

        for date in date_range:

            feature_columns = [column for column in prediction_df.columns if column != 'Volume']
            features = prediction_df[feature_columns].loc[date].copy()

            x_ = compose_forecast_frame(self.df_.to_numpy(), features.to_numpy(), self.lag)
            y_ = self.model_.predict(x_)

            features['Volume'] = y_
            self.df_ = pd.concat([self.df_, features.to_frame().T])

        return self.df_['Volume'].loc[date_range[0]:date_range[-1]]

    def dump(self, path: str) -> None:
        joblib.dump(self.model_, f"{path}/Volume_predictor.joblib")

    def load(self, df: pd.DataFrame, path: str):
        self.df_ = self.build(df.copy())
        self.model_ = joblib.load(f"{path}/Volume_predictor.joblib")


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
    indicators: List[str] = ['EMA20', 'RSI14', 'ATR14', 'MACD', 'MACD_Signal', 'MACD_Hist']

    def __init__(self, lag: int = 25):
        self.lag = lag
        self.feature_models = {
            'High': SLFM('High'),
            'Low': SLFM('Low'),
            'Volume': VolumeFM()
        }

    def build(self, df: pd.DataFrame):
        """
        Builds dataset from the dataframe.

        Parameters:
            df (pd.DataFrame): dataframe from which dataset is built.

        Returns:
            pd.DataFrame: prepared dataset.
        """

        # First difference to remove trend
        self.last_close_price = df.iloc[-1]['Close']
        df['Close'] = df['Close'].diff()
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

        # Predict feature targets
        for feature, feature_model in self.feature_models.items():
            df[feature] = self.feature_models[feature].predict(df.index)
        # Set `Open` price
        df['Open'] = np.nan
        df['Open'].iloc[0] = self.df_['Close'].iloc[-1]

        # Set `Close` price
        df['Close'] = np.nan

        # Set Indicators
        for indicator in self.indicators:
            df[indicator] = np.nan
            df[indicator].iloc[0] = self.df_[indicator].iloc[-1]

        return df

    def fit(self, df: pd.DataFrame):
        self.df_ = self.build(df.copy())

        # Fit feature models
        for feature_model in self.feature_models.values():
            feature_model.fit(self.df_)

        X_train, y_train = decompose_tabular_data(self.df_.to_numpy(), h=self.lag)

        train_size = int(len(self.df_) * 0.8)
        train_set, test_set = self.df_.iloc[:train_size], self.df_.iloc[train_size:]

        X_train, y_train = decompose_tabular_data(train_set.to_numpy(), h=self.lag)
        X_test, y_test = decompose_tabular_data(test_set.to_numpy(), h=self.lag)

        self.model_ = XGBRegressor(
            base_score=.5,
            booster='gbtree',
            early_stopping_rounds=10,
            objective='reg:squarederror',
            max_depth=3,
            learning_rate=.05
        )

        self.model_.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)

        return self

    def predict(self, date_range: pd.DatetimeIndex) -> pd.Series:

        prediction_df = self.build_forecast_data(pd.DataFrame(index=date_range))

        for idx, date in enumerate(date_range):

            feature_columns = [column for column in prediction_df.columns if column != 'Close']
            features = prediction_df[feature_columns].loc[date].copy()

            x_ = compose_forecast_frame(self.df_.to_numpy(), features.to_numpy(), self.lag)
            y_ = self.model_.predict(x_)

            history_close_price = pd.concat([self.df_['Close'], pd.Series(y_, index=[date])])
            history_high_price = pd.concat([self.df_['High'], pd.Series(features['High'], index=[date])])
            history_low_price = pd.concat([self.df_['Low'], pd.Series(features['Low'], index=[date])])

            # Indicators
            features['EMA20'] = ta.trend.EMAIndicator(history_close_price, window=20).ema_indicator().iloc[-1]
            features['RSI14'] = ta.momentum.RSIIndicator(history_close_price, window=14).rsi().iloc[-1]
            features['ATR14'] = ta.volatility.AverageTrueRange(
                history_high_price, history_low_price,
                history_close_price, window=14
            ).average_true_range().iloc[-1]

            macd = ta.trend.MACD(history_close_price, window_slow=26, window_fast=12, window_sign=9)
            features['MACD'] = macd.macd().iloc[-1]
            features['MACD_Signal'] = macd.macd_signal().iloc[-1]
            features['MACD_Hist'] = macd.macd_diff().iloc[-1]

            # Set future indicator values as current
            if idx < len(date_range)-1:
                for indicator in self.indicators:
                    prediction_df.loc[date_range[idx+1], indicator] = features[indicator]

                # Set `Open` price
                prediction_df.loc[date_range[idx+1], 'Open'] = self.df_['Close'].iloc[-1]

            features['Close'] = y_
            self.df_ = pd.concat([self.df_, features.to_frame().T])

        # Recover original time series
        y_pred = self.df_['Close'].loc[date_range[0]:date_range[-1]]
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
