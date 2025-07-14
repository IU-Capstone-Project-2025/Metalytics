#!/usr/bin/env python3
# zinc_features_yf_fixed_csv.py
# -----------------------------
# Скачиваем котировки цинка, DXY и Brent через yfinance (без API-ключей)
# и сохраняем в CSV.

import logging
from pathlib import Path
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")

TICKERS = {
    "ZINC" : "ZNC=F",     # 3-месячный futures на цинк (если пусто — замените)
    "DXY"  : "DX-Y.NYB",  # ICE Dollar Index
    "BRENT": "BZ=F",      # Brent front-month
}

OUTFILE = Path("zinc_yf_features.csv")


def fetch_series(ticker: str, col_name: str) -> pd.Series:
    """Скачивает Close-цены тикера через yfinance и возвращает Series."""
    try:
        df = yf.download(ticker, progress=False)
    except Exception as err:
        logging.error("Ошибка yfinance для %s: %s", ticker, err)
        return pd.Series(name=col_name, dtype="float64")

    if df.empty:
        logging.warning("Yahoo не вернул данных для %s (%s)", col_name, ticker)
        return pd.Series(name=col_name, dtype="float64")

    ser = df["Close"].copy()
    ser.name = col_name
    logging.info("Fetched %s — %d rows", col_name, len(ser))
    return ser


def main():
    # качаем базовые ряды
    frames = [fetch_series(tkr, name) for name, tkr in TICKERS.items()]
    data = pd.concat(frames, axis=1).sort_index()

    # считаем доходности и 5-дн. скользящее среднее
    for col in TICKERS.keys():
        if col in data:
            data[f"{col}_ret"] = data[col].pct_change()
            data[f"{col}_ma5"] = data[col].rolling(5).mean()

    # убираем строки, где вообще нет данных
    data = data.dropna(how="all")

    # сохраняем в CSV
    data.to_csv(OUTFILE, index_label="Date")
    logging.info("Saved %d rows × %d cols → %s", *data.shape, OUTFILE)


if __name__ == "__main__":
    main()
