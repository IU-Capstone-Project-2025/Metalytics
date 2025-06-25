import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

ticker = "GC=F"
interval = "1h"
save_path = "gold_futures_yahoo_1h.csv"


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
    print(f"⬇️ Загружаю {start.date()} ➡️ {end.date()} ...")
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if not df.empty:
        all_data.append(df)


if all_data:
    full_df = pd.concat(all_data)
    full_df = full_df[~full_df.index.duplicated()]
    full_df.sort_index(inplace=True)
    full_df.to_csv(save_path)
    print(f" Данные сохранены в {save_path}")
else:
    print(" Не удалось загрузить данные.")
