from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import fastapi
import uvicorn
import numpy
import sklearn
import pandas
import pkg_resources



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Hello!"}

@app.get("/metals")
def metals_check():
    '''
        Get a list of available metals (["gold"])
    '''
    return {"available_metals": ["gold"]}


@app.get("/forecast/{metal_id}")
def metal_forecast(metal_id: str):
    '''
        Get a metal price forecast
    '''
    return {"message":"Hello world"}


@app.get("/forecast/{metal_id}/days")
def metal_forcast_N_days(metal_id: str, num_days: int):
    '''
        Get prices for N days ahead
    '''
    return {"message":"Hello world"}

@app.get("/health")
def health_check():
    '''
        Health check endpoint to verify that backend is running
    '''
    return{"backend status":"OK"}

@app.get("/version")
def get_version():
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
            "docker": get_package_version("docker")
        }
    }


@app.get("/logs")
def get_logs():
    '''
        Output of logs
    '''
    return{"message":"Hello world"}

def get_package_version(package_name):
    try:
        return pkg_resources.get_distribution(package_name).version
    except Exception:
        return "N/A"
