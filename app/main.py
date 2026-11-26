"""HTTP interface for the bus-delay model."""

import json
import logging
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.bus_delay.data import FEATURES

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    route_km: float = Field(ge=1.5)
    stops: int = Field(le=70)
    passengers: int
    hour: int
    temp_c: float
    driver_exp_yr: float
    weather: str
    depot_zone: str


class PredictionResponse(BaseModel):
    prediction: float


class PredictionLogFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": self.formatTime(record),
            "event": record.getMessage(),
            "prediction_min": record.prediction_min,
        })


prediction_logger = logging.getLogger("mp001.prediction")
prediction_logger.setLevel(logging.INFO)
prediction_logger.propagate = False
if not prediction_logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(PredictionLogFormatter())
    prediction_logger.addHandler(handler)


@lru_cache(maxsize=1)
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail="Champion model is unavailable") from error


app = FastAPI(title="Lucknow Bus Delay")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> PredictionResponse:
    model = load_model()
    features = pd.DataFrame([request.model_dump()], columns=FEATURES)
    prediction = float(model.predict(features)[0])
    prediction_logger.info("bus_delay_prediction", extra={"prediction_min": prediction})
    return PredictionResponse(prediction=prediction)
