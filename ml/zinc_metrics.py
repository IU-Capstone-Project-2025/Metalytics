#!/usr/bin/env python3
# zinc_macro_features_final.py
# Robust open‑data collector (July‑2025)

import logging
from io import StringIO
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)


# ---------- helper: Yahoo series --------------------------------
def yf_close(ticker: str, name: str) -> pd.Series:
    """Fetch a closing price series from Yahoo Finance."""
    try:
        df = yf.download(
            ticker, period="10y", interval="1d", progress=False, auto_adjust=False
        )

        if df is None or df.empty:
            logging.warning(
                "yfinance %s (%s) returned an empty or None DataFrame.", ticker, name
            )
            return pd.Series(name=name, dtype="float64")

        close_series = None
        # yfinance sometimes returns multi-level columns, so we handle both cases.
        if "Close" in df.columns:
            s = df["Close"]
            if isinstance(s, pd.DataFrame):
                # Multi-level columns: the result is a DataFrame with one column.
                if not s.empty:
                    close_series = s.iloc[:, 0]
            elif isinstance(s, pd.Series):
                # Single-level columns: we get a Series directly.
                close_series = s

        if close_series is not None:
            result = close_series.copy()
            result.name = name
            return result

        logging.warning("Could not extract 'Close' series for %s (%s).", ticker, name)
        logging.info("DataFrame columns for %s (%s): %s", ticker, name, df.columns)
        return pd.Series(name=name, dtype="float64")

    except Exception as e:
        logging.error("yfinance %s (%s) failed with exception: %s", ticker, name, e)
        return pd.Series(name=name, dtype="float64")


# ---------- helper: FRED data ------------------------------------
def fred_csv(series_id: str, name: str) -> pd.Series:
    """Fetch a series from FRED as a CSV."""
    logging.info("Fetching %s from FRED (%s)", name, series_id)
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        r = requests.get(url, timeout=18)
        if r.status_code != 200:
            logging.debug("FRED CSV %s HTTP %s", series_id, r.status_code)
            return pd.Series(name=name, dtype="float64")

        csv_data = StringIO(r.text)
        # The date column name can vary ('DATE' or 'observation_date'),
        # so we use the first column as the index to be more robust.
        df = pd.read_csv(csv_data, index_col=0, parse_dates=True)

        s = df[series_id].copy()
        if isinstance(s, pd.Series):
            s.name = name
            s.replace(".", pd.NA, inplace=True)
            s = s.ffill()
            s = s.astype(float)
            return s

        logging.warning("FRED data for %s is not a Series.", name)
        return pd.Series(name=name, dtype="float64")

    except ValueError as e:
        logging.error("Failed to parse FRED CSV for %s due to ValueError: %s", name, e)
        if "r" in locals() and r is not None:
            logging.error("Response text from FRED that caused the error: %s", r.text)
        return pd.Series(name=name, dtype="float64")
    except Exception as e:
        logging.error(
            "Failed to parse FRED CSV for %s with a general exception: %s",
            name,
            e,
            exc_info=True,
        )
        return pd.Series(name=name, dtype="float64")


# ---------- Data Fetching ---------------------------------------

# Financial data from Yahoo Finance
zn_price = yf_close("ZNC=F", "Zn_Price")
dxy = yf_close("DX-Y.NYB", "DXY")
brent = yf_close("BZ=F", "Brent_USD")

# Macroeconomic data from FRED
# The previous PMI series was unreliable; switching to the ISM series.
global_pmi = fred_csv("ISMIR", "Global_PMI")
zinc_production = fred_csv("IPG21223SQ", "Zinc_Production")


# ---------- Assemble Dataframe ----------------------------------
all_series = [
    zn_price,
    dxy,
    brent,
    zinc_production,
    global_pmi,
]
df = pd.concat(all_series, axis=1).sort_index()

if df.empty:
    logging.error("No data collected! Check network or proxies.")
else:
    # Filter for the last 10 years to align all data series
    ten_years_ago = pd.to_datetime("today") - pd.DateOffset(years=10)
    df = df[df.index >= ten_years_ago]

    df = df.asfreq("B")

    # Backward-fill the scraped single-point values
    scraped_cols = ["Zinc_Production", "Global_PMI"]
    for col in scraped_cols:
        if col in df.columns:
            series = df[col]
            if isinstance(series, pd.Series):
                df[col] = series.bfill().ffill()  # bfill then ffill for good measure

    logging.info("DataFrame head after filling:\n%s", df.head().to_string())
    logging.info("DataFrame tail after filling:\n%s", df.tail().to_string())

    # Interpolate the financial time series
    df.interpolate(method="time", inplace=True)

    df.to_csv("zinc_macro_features.csv", index_label="Date", float_format="%.5f")
    logging.info(
        "Saved %d rows × %d cols → zinc_macro_features.csv",
        len(df),
        len(df.columns),
    )
