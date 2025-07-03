import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import ta


class DatasetLoaderInterface:
    def load(self) -> pd.DataFrame:
        raise NotImplementedError()


class GoldDatasetLoader(DatasetLoaderInterface):
    def __init__(self,
                 symbol="GC=F",
                 interval="1h",
                 csv_path="gold_futures_with_indicators.csv",
                 raw_path="gold_futures_yahoo_1h.csv"):
        self.symbol = symbol
        self.interval = interval
        self.csv_path = csv_path
        self.raw_path = raw_path
        self._df_cache = None

    def load(self, force_update=False) -> pd.DataFrame:
        if not force_update and self._df_cache is not None:
            return self._df_cache

        if not force_update and os.path.exists(self.csv_path) and not self.needs_update():
            self._df_cache = pd.read_csv(self.csv_path, index_col=0, parse_dates=True)
            return self._df_cache

        print("Обновляю данные")
        self._download_raw_data()
        self._calculate_indicators()
        self._df_cache = pd.read_csv(self.csv_path, index_col=0, parse_dates=True)
        return self._df_cache

    def needs_update(self) -> bool:
        if not os.path.exists(self.csv_path):
            return True
        df = pd.read_csv(self.csv_path, index_col=0, parse_dates=True)
        if df.empty:
            return True
        last_date = df.index[-1]
        now = pd.Timestamp.utcnow()
        return (now - last_date) > pd.Timedelta(days=1)

    def _download_raw_data(self):
        STEP_DAYS = 60
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=730)

        date_ranges = []
        cursor = start_date
        while cursor < end_date:
            next_cursor = min(cursor + timedelta(days=STEP_DAYS), end_date)
            date_ranges.append((cursor, next_cursor))
            cursor = next_cursor

        all_data = []
        for start, end in date_ranges:
            df = yf.download(self.symbol, start=start, end=end, interval=self.interval, progress=False)
            if not df.empty:
                all_data.append(df)

        if all_data:
            full_df = pd.concat(all_data)
            full_df = full_df[~full_df.index.duplicated()]
            full_df.sort_index(inplace=True)
            full_df.to_csv(self.raw_path)
        else:
            raise RuntimeError("Не удалось загрузить данные")

    def _calculate_indicators(self):
        df = pd.read_csv(self.raw_path, skiprows=3, index_col=0, parse_dates=True)
        df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['Close', 'High', 'Low', 'Open'], inplace=True)

        df['EMA20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
        df['RSI14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        df['ATR14'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
        macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()

        df.to_csv(self.csv_path)
