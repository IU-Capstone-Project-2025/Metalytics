import pandas as pd
from zinc_parsing import zinc_parsing
from zinc_metrics import zinc_metrics
from forecasting_models import ZincForecastModel

def update_model_zinc():
    zinc_parsing()
    zinc_metrics()
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


    # 1) обучаем
    zm = ZincForecastModel(window_size=48, split_date="2024-07-01")
    zm.fit(data_df, params=dict(epochs=40, batch_size=128))

    # 2) сохраняем
    zm.dump("./lstm_zinc")

    # # 3) в другом ноутбуке / после перезапуска
    # zm2 = ZincForecastModel()
    # zm2.load(data_df, "./lstm_zinc")