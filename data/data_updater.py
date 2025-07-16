import time
from fetch_gold_ohlcv import fetch_gold_ohlcv
from calculate_indicators import calculate_indicators
from insert_forecast_to_db import insert_forecast_to_db
from update_model import update_model
from parse_metallurgy_news import parse_metallurgy_news

hour_counter = 0

while True:
    fetch_gold_ohlcv()
    calculate_indicators()
    if hour_counter == 0:
        parse_metallurgy_news()
        update_model()
    hour_counter = (hour_counter + 1) % 24
    insert_forecast_to_db()
    time.sleep(3600)
