from insert_forecast_to_db import update_predicted_prices
from forecasting_models import ZincForecastModel
import pandas as pd
import os
from dotenv import load_dotenv


def insert_forecast_zinc_to_db():
    # 1.1  OHLCV‑файл (почасовой тайм‑фрейм)
    ohlcv_df = pd.read_csv(
        "zinc_ohlcv_safe.csv",
        header=None, skiprows=3,
        names=["DateTime", "Open", "High", "Low", "Close", "Volume"],
    )
    ohlcv_df["DateTime"] = pd.to_datetime(ohlcv_df["DateTime"])
    ohlcv_df["Date"]     = ohlcv_df["DateTime"].dt.date   # удобное поле для merge

    # 1.2  Макро‑фичи (дневной файл)
    macro_df = pd.read_csv("zinc_macro_features.csv")
    macro_df["Date"] = pd.to_datetime(macro_df["Date"]).dt.date

    # 1.3  Объединяем → сортируем → заполняем пропуски
    data_df = (ohlcv_df
            .merge(macro_df, on="Date", how="left")
            .sort_values("DateTime")
            .fillna(method="ffill"))
    
    print("#"*50,"data_df","#"*50)
    print(data_df)

    zm2 = ZincForecastModel()
    zm2.load(data_df, "./lstm_zinc")
    
    future_idx = pd.date_range(data_df['DateTime'].iloc[-1], periods=48, freq="h")
    pred_24h = zm2.predict(future_idx)

    print("#"*50,"pred_24h","#"*50)
    print(pred_24h)

    load_dotenv()
    # Параметры подключения к БД (замените на свои)
    db_params = {
        'dbname': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT')
    }

    # Вызываем функцию для обновления данных
    update_predicted_prices(pred_24h, 3, db_params)


if __name__ == "__main__":
    insert_forecast_zinc_to_db()

