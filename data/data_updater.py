import time
from fetch_gold_ohlcv import fetch_gold_ohlcv
from calculate_indicators import calculate_indicators
from insert_forecast_to_db import insert_forecast_to_db
from update_model import update_model
from parse_metallurgy_news import parse_metallurgy_news
import datetime as dt

# Take today as yesterday to have 
# first data updating
today = dt.date.today() - dt.timedelta(days=1)

metal_list = [
    'gold',
    'silver'
]

while True:
    if today != dt.date.today():
        # parse_metallurgy_news()
        update_model()
        today = dt.date.today()

    for metal_id in metal_list:
        insert_forecast_to_db(metal_id)
    time.sleep(3600)
