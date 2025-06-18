from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "Hello!"}

@app.get("/metals")
async def metals_check():
    '''
        Get a list of available metals (["gold", "nickel", "aluminum"])
    '''
    return {"message":"Hello world"}

@app.get("/forecast/{metal_id}")
async def metal_forecast(metal_id: str):
    '''
        Get a metal price forecast
    '''
    return {"message":"Hello world"}


@app.get("/forecast/{metal_id}/days")
async def metal_forcast_N_days(metal_id: str, num_days: int):
    '''
        Get prices for N days ahead
    '''
    return {"message":"Hello world"}

@app.get("/health")
async def health_check():
    '''
        Health check endpoint to verify that backend is running
    '''
    return{"message":"Hello world"}

@app.get("/version")
async def get_version():
    '''
    Returns current API and module versions
    '''
    return {
        "api_version": "0.1.0",
        "module_versions": {
            "pandas": "N/A",
            "scikit-learn": "N/A",
            "numpy": "N/A"
        }
    }

@app.get("/logs")
async def get_logs():
    '''
        Output of logs
    '''
    return{"message":"Hello world"}
