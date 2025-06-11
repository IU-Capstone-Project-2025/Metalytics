## Scripts
- `fetch_gold_ohlcv.py`: script for collecting OHLCV data (Open, High, Low, Close, Volume).
- `calculate_indicators.py`: script for calculating indicators (EMA20, RSI14, ATR14, MACD) based on the collected data.

### Dependecies Installation
1. Prerequisite version of Python >= 3.7.
2. Dependencies installation via pip: 
`pip install pandas yfinance ta`
3. Validate the installation by running the following commands in Python:
```
import pandas
import yfinance
import ta
```

### Usage
1. Run `fetch_gold_ohlcv.py` for collecting data into `gold_futures_yahoo_1h.csv`.
2. Run `calculate_indicators.py` for adding indicators and saving into `gold_futures_with_indicators.csv`.

## Jupyter Notebooks
Jupyter notebooks contain data exploration, task-specific analysis, and model selection. The following notebooks are uploaded:
- `01_filter_design.ipynb`: data visual analysis, baseline model implementation, and filtration design.
