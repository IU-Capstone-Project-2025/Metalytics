from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import os
import psutil
import pkg_resources
import fastapi
import uvicorn
import numpy
import sklearn
import pandas


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


@app.get("/")
async def read_root():
    return {"message": "Hello!"}


@app.get("/metals")
async def metals_check():
    '''
        Get a list of available metals (["gold"])
    '''
    return {"available_metals": ["gold"]}


@app.get("/forecast/{metal_id}")
async def metal_forecast(metal_id: str):
    '''
        Get a metal price forecast
    '''
    return {"message": "Hello world"}


@app.get("/forecast/{metal_id}/days")
async def metal_forcast_N_days(metal_id: str, num_days: int):
    '''
        Get prices for N days ahead
    '''
    return {"message": "Hello world"}


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


@app.get("/logs")
async def get_logs():
    '''
        Output of logs
    '''
    return {"message": "Hello world"}


def get_package_version(package_name: str) -> str:
    try:
        return pkg_resources.get_distribution(package_name).version
    except Exception:
        return "N/A"
