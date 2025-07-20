from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_metals():
    response = client.get("/metals")
    assert response.status_code == 200
    assert "available_metals" in response.json()


def test_historical_data():
    response = client.get("/historical_data/Gold?period=1mo&interval=1d")
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        assert isinstance(response.json(), list)


# def test_forecast():
#     response = client.get("/forecast/Gold")
#     assert response.status_code in (200, 404)
#     if response.status_code == 200:
#         assert isinstance(response.json(), list)


# def test_forecast_days():
#     response = client.get("/forecast/Gold/days?unit=h&value=24")
#     assert response.status_code in (200, 404)
#     if response.status_code == 200:
#         assert isinstance(response.json(), list)
