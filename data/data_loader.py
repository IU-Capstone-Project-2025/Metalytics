import pandas as pd
import yfinance as yf
import ta
import numpy as np
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
        else:
            print("Warning: No gold data fetched.")
            full_df = pd.DataFrame()
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
            print("Updating the gold data")
            raw_df = self._fetch_raw_data()
            if raw_df.empty:
                print("Error: No gold data available.")
                return pd.DataFrame()
            processed_df = self._calculate_indicators(raw_df)
            processed_df.to_csv(self.processed_data_path)
            self.data = processed_df
        else:
            print("Previously uploaded gold data is used")
            self.data = pd.read_csv(self.processed_data_path, index_col=0, parse_dates=True)
            self.data = self.data[~self.data.index.duplicated()]
        return self.data

class SilverDataLoader:
    def __init__(self, raw_data_path="silver_futures_yahoo_1h.csv", processed_data_path="silver_futures_with_indicators.csv"):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.last_update_date = None
        self.data = None

    def _fetch_raw_data(self):
        """Downloads raw data for silver and S&P 500 Close."""
        tickers = ["SI=F", "^GSPC"]  # Silver and S&P 500
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

        all_data = {ticker: [] for ticker in tickers}
        for start, end in date_ranges:
            for ticker in tickers:
                print(f"Fetching {ticker} data from {start} to {end}")
                df = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
                if not df.empty:
                    # Flatten MultiIndex columns if present
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = ['_'.join(col).strip() for col in df.columns]
                    # Select only 'Close' for ^GSPC, drop 'Adj Close' for SI=F
                    if ticker == "^GSPC":
                        df = df[['Close_^GSPC']].rename(columns={'Close_^GSPC': 'SP500'})
                    else:  # SI=F
                        df = df[['Close_SI=F', 'High_SI=F', 'Low_SI=F', 'Open_SI=F', 'Volume_SI=F']].rename(
                            columns={
                                'Close_SI=F': 'Close',
                                'High_SI=F': 'High',
                                'Low_SI=F': 'Low',
                                'Open_SI=F': 'Open',
                                'Volume_SI=F': 'Volume'
                            }
                        )
                    print(f"Fetched {ticker} shape: {df.shape}, columns: {df.columns.tolist()}")
                    all_data[ticker].append(df)
                else:
                    print(f"Warning: No data fetched for {ticker} from {start} to {end}")

        # Combine data for each ticker
        combined_data = []
        for ticker in tickers:
            if all_data[ticker]:
                df = pd.concat(all_data[ticker])
                df = df[~df.index.duplicated(keep='first')]
                df.sort_index(inplace=True)
                if ticker == "SI=F":
                    print(f"Silver data shape: {df.shape}, non-NaN Close counts: {df['Close'].count()}")
                else:  # ^GSPC
                    print(f"S&P 500 data shape: {df.shape}, non-NaN SP500 counts: {df['SP500'].count()}")
                combined_data.append(df)
            else:
                print(f"Error: No data available for {ticker}")
                if ticker == "^GSPC":
                    # Create a dummy SP500 column with NaN if no data
                    df = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq='1H'))
                    df['SP500'] = np.nan
                    print(f"Dummy S&P 500 data shape: {df.shape}")
                    combined_data.append(df)

        # Merge silver and S&P 500 data
        if combined_data:
            full_df = combined_data[0]  # Silver data
            for df in combined_data[1:]:
                full_df = full_df.join(df, how='left')  # Left join to keep silver data
            print(f"Combined DataFrame columns: {full_df.columns.tolist()}")
            print(f"Combined DataFrame shape: {full_df.shape}, non-NaN SP500 counts: {full_df['SP500'].count() if 'SP500' in full_df.columns else 0}")
            print(f"Combined DataFrame head:\n{full_df.head()}")
            full_df.to_csv(self.raw_data_path)
        else:
            print("Error: No data fetched for any ticker.")
            full_df = pd.DataFrame()
        return full_df

    def _calculate_indicators(self, df):
        """Adds indicators and handles S&P 500"""
        expected_columns = ['Close', 'High', 'Low', 'Open', 'Volume', 'SP500']
        for col in expected_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                print(f"Warning: Column {col} missing in DataFrame. Adding with NaN.")
                df[col] = np.nan

        df.dropna(subset=['Close', 'High', 'Low', 'Open'], inplace=True)
        # Interpolate SP500 to handle missing values
        if 'SP500' in df.columns:
            non_nan_before = df['SP500'].count()
            df['SP500'] = df['SP500'].interpolate(method='time', limit_direction='both')
            non_nan_after = df['SP500'].count()
            print(f"SP500 non-NaN counts before interpolation: {non_nan_before}, after: {non_nan_after}")
            print(f"SP500 sample values:\n{df['SP500'].head()}")
            if non_nan_after < len(df) * 0.1:
                print(f"Warning: SP500 has very few non-NaN values ({non_nan_after}/{len(df)}) after interpolation.")
        else:
            print("Warning: SP500 column missing after processing. Adding with NaN.")
            df['SP500'] = np.nan

        df['EMA20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
        df['RSI14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        df['ATR14'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()

        macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        print(f"Processed DataFrame columns: {df.columns.tolist()}")
        print(f"Processed DataFrame shape: {df.shape}, non-NaN SP500 counts: {df['SP500'].count() if 'SP500' in df.columns else 0}")
        print(f"Processed DataFrame head:\n{df.head()}")
        return df

    def _needs_update(self):
        """Force refresh to ensure SP500 is included"""
        return True  # Remove this line and uncomment below after confirming SP500 is included
        # if not os.path.exists(self.processed_data_path):
        #     return True
        # last_modified = datetime.fromtimestamp(os.path.getmtime(self.processed_data_path))
        # return (datetime.now() - last_modified) > timedelta(days=1)

    def load_data(self):
        """Main method: returns up-to-date data"""
        if self._needs_update():
            print("Updating the silver data")
            raw_df = self._fetch_raw_data()
            if raw_df.empty:
                print("Error: No silver data available.")
                return pd.DataFrame()
            processed_df = self._calculate_indicators(raw_df)
            processed_df.to_csv(self.processed_data_path)
            self.data = processed_df
        else:
            print("Previously uploaded silver data is used")
            self.data = pd.read_csv(self.processed_data_path, index_col=0, parse_dates=True)
            self.data = self.data[~self.data.index.duplicated()]
            if 'SP500' not in self.data.columns:
                print("Warning: SP500 column missing in loaded data. Adding with NaN.")
                self.data['SP500'] = np.nan
        print(f"Final DataFrame columns: {self.data.columns.tolist()}")
        print(f"Final DataFrame shape: {self.data.shape}, non-NaN SP500 counts: {self.data['SP500'].count() if 'SP500' in self.data.columns else 0}")
        print(f"Final DataFrame head:\n{self.data.head()}")
        return self.data