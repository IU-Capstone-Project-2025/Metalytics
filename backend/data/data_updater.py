import time
from fetch_gold_ohlcv import fetch_gold_ohlcv 
from calculate_indicators import calculate_indicators
from insert_forecast_to_db import insert_forecast_to_db
import os

while True:
    fetch_gold_ohlcv()
    calculate_indicators()
    insert_forecast_to_db()
    time.sleep(3600)