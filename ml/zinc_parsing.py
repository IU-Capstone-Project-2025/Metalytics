
import logging
from pathlib import Path
import yfinance as yf
import pandas as pd           

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

TICKER   = "ZNC=F"
OUTFILE  = Path("zinc_ohlcv_safe.csv")

def fetch_ohlc(tkr: str, interval: str) -> pd.DataFrame:
    """Запрашивает свечи указанного интервала через yfinance."""
    period = "730d" if interval != "1d" else "max"   
    return yf.download(tkr, interval=interval,
                       period=period,
                       progress=False, auto_adjust=False)

def main() -> None:
   
    df = fetch_ohlc(TICKER, "60m")
    if df.empty:
        logging.warning("Почасовых данных нет – берём дневные ('1d').")
        df = fetch_ohlc(TICKER, "1d")

    if df.empty:
        logging.error("Не удалось получить даже дневные данные для %s.", TICKER)
        return

    df = df.rename(columns=str.title)[['Open', 'High', 'Low', 'Close', 'Volume']]

    df.to_csv(
        OUTFILE,
        index_label="DateTime",
        float_format="%.4f",
        date_format="%Y-%m-%d %H:%M"
    )
    logging.info("Saved %d строк × %d колонок → %s", len(df), len(df.columns), OUTFILE)

if __name__ == "__main__":
    main()
