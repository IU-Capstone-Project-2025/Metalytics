import pandas as pd
import ta

input_path = "gold_futures_yahoo_1h.csv"
output_path = "gold_futures_with_indicators.csv"

df = pd.read_csv(input_path, skiprows=3, index_col=0, parse_dates=True)

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

df.to_csv(output_path)
print(f" Сохранение в {output_path}")
