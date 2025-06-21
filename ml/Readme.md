## Scripts
- `fetch_gold_ohlcv.py`: script for collecting OHLCV data (Open, High, Low, Close, Volume).
- `calculate_indicators.py`: script for calculating indicators (EMA20, RSI14, ATR14, MACD) based on the collected data.
- `forecasting_models.py`: script defining key models for target forecasting.
- `forecasting_framework.py`: script defining the class for working with models.
- `client.py`: script with example code of using forecasting framewrok

### Dependencies Installation
1. Prerequisite version of Python >= 3.7.
2. Dependencies installation via pip: 
`pip install -r requirements.txt`

### Usage
1. Run `fetch_gold_ohlcv.py` for collecting data into `gold_futures_yahoo_1h.csv`.
2. Run `calculate_indicators.py` for adding indicators and saving into `gold_futures_with_indicators.csv`.
3. Run `client.py` to test out model creation, training, dumping, and loading.

## Jupyter Notebooks
Jupyter notebooks contain data exploration, task-specific analysis, and model selection. The following notebooks are uploaded:
- `01_filter_design.ipynb`: data visual analysis, baseline model implementation, and filtration design.
