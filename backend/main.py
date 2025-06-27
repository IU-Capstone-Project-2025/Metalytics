from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from forecasting_framework import ForecastFramework
import yfinance as yf
import os
import psutil
import pkg_resources
import fastapi
import uvicorn
import numpy
import sklearn
import pandas as pd


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_system_metrics() -> Dict[str, Any]:
    """Optional function to gather system metrics"""
    try:
        return {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
        }
    except Exception:
        return {"error": "Could not gather system metrics"}

# Dictionary for metals from name to tags
Metal_dict = {
    "Gold": "GC=F",
}

@app.get("/")
async def read_root():
    return {"message": "Hello!"}


@app.get("/metals")
async def metals_check():
    '''
        Get a list of available metals (["gold"])
    '''
    return {"available_metals": [key for key in Metal_dict.keys()]}

@app.get("/historical_data/{metal_id}")
async def metal_historical_data(metal_id: str, period: str, interval = "1d"):
    '''
        Get a historical data. 
        - metal_id: Metal name - e.g. "Gold"
        - period: Time period - ["1h", "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]. 
        - interval: Data interval - ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
    '''
    try:
        is_period_1hour = False
        
        # Check input parameters
        if metal_id not in Metal_dict.keys():
            raise HTTPException(status_code=404, detail="No matches for this metal")

        if period not in ["1h", "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]:
            raise HTTPException(status_code=400, detail="Wrong period parameter")

        if interval not in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]:
            raise HTTPException(status_code=400, detail="Wrong period parameter")
        
        if period == "1h":
            is_period_1hour = True
            period = "1d"
        
        # Fetch metal data
        ticker_symbol = Metal_dict[metal_id]
        metal = yf.Ticker(ticker_symbol)
        
        # Get historical data
        hist = metal.history(
            period=period,
            interval=interval
        )
        
        if hist.empty:
            raise HTTPException(status_code=404, detail="No data found for the given parameters")

        # Convert DataFrame to dictionary
        formatted_data = []
        for date, row in hist.iterrows():
            # Convert numpy.float64 to native Python float
            formatted_data.append({
                "timestamp": date.isoformat() + "Z",
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"])
            })

        if is_period_1hour:
            # pattern = formatted_data[-1]['timestamp'][:13]

            # for i in range(1,61):
            #     if formatted_data[-i]['timestamp'][:13] != pattern:
            #         i-=1
            #         break

            # croped_data = formatted_data[-i:]

            croped_data = {"msg":"wrong interval selection"}
            
            if interval == '1m':
                croped_data = formatted_data[-60:]
            if interval == '2m':
                croped_data = formatted_data[-30:]
            if interval == '5m':
                croped_data = formatted_data[-12:]
            if interval == '15m':
                croped_data = formatted_data[-4:]
            if interval == '30m':
                croped_data = formatted_data[-2:]
            if interval == '60m':
                croped_data = formatted_data[-1:]
            return croped_data

        return formatted_data
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    


@app.get("/forecast/{metal_id}")
async def metal_forecast(metal_id: str):
    '''
        Get a metal price forecast for 1 day
        - metal_id: Metal name - e.g. "Gold"
    '''
    
    try:

        # Check input parameters
        if metal_id not in Metal_dict.keys():
            raise HTTPException(status_code=404, detail="No data found for the given parameters")

        dataframe = pd.read_csv("data/gold_futures_with_indicators.csv", parse_dates=[0], index_col=0)

        
        # Load existing model
        path: str = "baseline_model"
        fm = ForecastFramework.load_from_file(path, dataframe)

        # Create forecast
        unit = 'h'  # units of time (e.g. 'h' for hour, 'd' for days, 'm' for months)
        value = 24   # value of units

        # Obtain pandas series with forecasted data
        forecast = fm.create_forecast(value=value, unit=unit)
        if forecast.empty:
            raise HTTPException(status_code=404, detail="No forecast. Something goes wrong")

        formatted_data = []
        for column_name in forecast.index:
            time, stamp = str(column_name).split()
            timestamp = time + "T" + stamp + "Z"
            price = float(forecast[column_name])
            formatted_data.append({
                "timestamp": timestamp,
                "price": price
            })
        
        return formatted_data
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    


@app.get("/forecast/{metal_id}/days")
async def metal_forcast_value_of_units(metal_id: str, unit = "h", value = 24):
    '''
        Get a metal price forecast for value number of selected unit
        - metal_id: Metal name - e.g. "Gold"
        - unit: hour, day or months - ['h', 'd', 'm']
        - value: value of units
    '''
    value = int(value)
    print (f"value: {value}")
    try:

        # Check input parameters
        if metal_id not in Metal_dict.keys():
            raise HTTPException(status_code=404, detail="No data found for the given parameters")
        
        if unit not in ['h', 'd', 'm']:
            raise HTTPException(status_code=400, detail="Wrong unit parameter")
        

        dataframe = pd.read_csv("data/gold_futures_with_indicators.csv", parse_dates=[0], index_col=0)
        
        # Load existing model
        path: str = "baseline_model"
        fm = ForecastFramework.load_from_file(path, dataframe)

        # Obtain pandas series with forecasted data
        forecast = fm.create_forecast(value=value, unit=unit)
        if forecast.empty:
            raise HTTPException(status_code=404, detail="No forecast. Something went wrong")

        formatted_data = []
        for column_name in forecast.index:
            time, stamp = str(column_name).split()
            timestamp = time + "T" + stamp + "Z"
            price = float(forecast[column_name])
            formatted_data.append({
                "timestamp": timestamp,
                "price": price
            })
        
        return formatted_data
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health_check(
    include_system_metrics: Optional[bool] = Query(False),
    include_env: Optional[bool] = Query(False)
) -> Dict[str, Any]:
    '''
        Health check endpoint to verify that backend is running
    '''
    status_info = {
        "status": "OK",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "Metalytics",
        "version": "1.0.0",
    }

    if include_system_metrics:
        status_info["system"] = get_system_metrics()

    if include_env:
        status_info["environment"] = {
            k: v for k, v in os.environ.items()
            if not k.upper().startswith(("SECRET", "PASSWORD", "KEY"))
        }

    return status_info


@app.get("/version")
async def get_version():
    """
    Returns current API and module versions
    """
    return {
        "api_version": "0.1.0",
        "module_versions": {
            "fastapi": get_package_version("fastapi"),
            "uvicorn": get_package_version("uvicorn"),
            "numpy": get_package_version("numpy"),
            "scikit-learn": get_package_version("scikit-learn"),
            "pandas": get_package_version("pandas"),
            "docker": get_package_version("docker"),
        }
    }


def get_package_version(package_name: str) -> str:
    try:
        return pkg_resources.get_distribution(package_name).version
    except Exception:
        return "N/A"

# # delete it before push
# if __name__ == "__main__":
#     uvicorn.run("main:app")
