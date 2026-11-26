import json
import sys

import pytest
from fastapi.testclient import TestClient

from app import main
from src.bus_delay.data import FEATURES


@pytest.fixture
def request_body():
    return {
        "route_km": 1.5,
        "stops": 70,
        "passengers": 40,
        "hour": 8,
        "temp_c": 28.0,
        "driver_exp_yr": 6.0,
        "weather": "clear",
        "depot_zone": "North",
    }


def test_health():
    response = TestClient(main.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_accepts_boundary_inputs_and_logs_json(request_body, monkeypatch, capsys):
    class Model:
        def predict(self, frame):
            assert list(frame) == FEATURES
            assert frame.iloc[0].to_dict() == request_body
            return [12.25]

    monkeypatch.setattr(main, "load_model", Model)
    monkeypatch.setattr(main.prediction_logger.handlers[0], "stream", sys.stdout)
    response = TestClient(main.app).post("/predict", json=request_body)
    assert response.status_code == 200
    assert response.json() == {"prediction": 12.25}
    line = json.loads(capsys.readouterr().out.strip())
    assert line["event"] == "bus_delay_prediction"
    assert line["prediction_min"] == 12.25


@pytest.mark.parametrize("bad_values", [{"stops": 71}, {"route_km": 1.49}, {"record_id": "extra"}])
def test_invalid_requests_do_not_load_model(request_body, bad_values, monkeypatch):
    def unexpected_load():
        pytest.fail("Invalid request reached the model")

    monkeypatch.setattr(main, "load_model", unexpected_load)
    response = TestClient(main.app).post("/predict", json={**request_body, **bad_values})
    assert response.status_code == 422
