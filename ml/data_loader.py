import pandas as pd
import yfinance as yf
import ta
from datetime import datetime, timedelta
import os

class GoldDataLoader:
    def __init__(self, raw_data_path="gold_futures_yahoo_1h.csv", processed_data_path="gold_futures_with_indicators.csv"):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.last_update_date = None
        self.data = None

    def _fetch_raw_data(self):
        """Downloads raw data"""
        ticker = "GC=F"
        interval = "1h"
        STEP_DAYS = 60

        end_date = datetime.today()
        start_date = end_date - timedelta(days=730)

        date_ranges = []
        cursor = start_date
        while cursor < end_date:
            next_cursor = min(cursor + timedelta(days=STEP_DAYS), end_date)
            date_ranges.append((cursor, next_cursor))
            cursor = next_cursor

        all_data = []
        for start, end in date_ranges:
            df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
            if not df.empty:
                all_data.append(df)

        if all_data:
            full_df = pd.concat(all_data)
            full_df = full_df[~full_df.index.duplicated()]
            full_df.sort_index(inplace=True)
            full_df.to_csv(self.raw_data_path)
        return full_df

    def _calculate_indicators(self, df):
        """Adds indicators"""
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
        return df

    def _needs_update(self):
        """Checks whether the data needs to be updated (if >1 day has passed since the last update)"""
        if not os.path.exists(self.processed_data_path):
            return True
        last_modified = datetime.fromtimestamp(os.path.getmtime(self.processed_data_path))
        return (datetime.now() - last_modified) > timedelta(days=1)

    def load_data(self):
        """Main method: returns up-to-date data"""
        if self._needs_update():
            print("Updating the data")
            raw_df = self._fetch_raw_data()
            processed_df = self._calculate_indicators(raw_df)
            processed_df.to_csv(self.processed_data_path)
            self.data = processed_df
        else:
            print("Previously uploaded data is used")
            self.data = pd.read_csv(self.processed_data_path, index_col=0, parse_dates=True)
        return self.data
