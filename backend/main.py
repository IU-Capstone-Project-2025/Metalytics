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
def read_root():
    return {"message": "Hello!"}

@app.get("/metals")
def metals_check():
    '''
        Get a list of available metals (["gold", "nickel", "aluminum"])
    '''
    return {"message":"Hello world"}

@app.get("/forecast/{metal_id}")
def metal_forecast(metal_id: str):
    '''
        Get a metal price forecast
    '''
    return {"message":"Hello world"}


@app.get("/forecast/{metal_id}?days=30")
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
    return{"message":"Hello world"}

@app.get("/version")
def get_version():
    '''
        Returns current API and module versions
    '''
    return{"message":"Hello world"}

@app.get("/logs")
def get_logs():
    '''
        Output of logs (for debugging)
    '''
    return{"message":"Hello world"}
