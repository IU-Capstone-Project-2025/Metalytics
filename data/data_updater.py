import time
from fetch_gold_ohlcv import fetch_gold_ohlcv
from calculate_indicators import calculate_indicators
from insert_forecast_to_db import insert_forecast_to_db
from update_model import update_model
from parse_metallurgy_news import parse_metallurgy_news

hour_counter = 0

metal_list = [
    'gold',
    'silver'
]

while True:
    if hour_counter == 0:
        parse_metallurgy_news()
        update_model()
    hour_counter = (hour_counter + 1) % 24
    for metal_id in metal_list:
        insert_forecast_to_db(metal_id)
    time.sleep(3600)
