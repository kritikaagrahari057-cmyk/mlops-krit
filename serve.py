import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
FEATURES = ["sqft", "bedrooms", "bathrooms", "age_years", "garage", "location_score"]

mlflow.set_tracking_uri("sqlite:///" + str(BASE_DIR / "mlflow.db"))
MODEL_URI = os.getenv("MODEL_URI", "models:/house-price-predictor@champion")
model = mlflow.sklearn.load_model(MODEL_URI)

app = FastAPI(title="House Price Predictor")


class HouseFeatures(BaseModel):
    sqft: float = Field(..., gt=0, le=20000)
    bedrooms: int = Field(..., ge=0, le=20)
    bathrooms: int = Field(..., ge=0, le=20)
    age_years: int = Field(..., ge=0, le=100)
    garage: int = Field(..., ge=0, le=10)
    location_score: int = Field(..., ge=1, le=10)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: HouseFeatures):
    row = pd.DataFrame([features.model_dump()], columns=FEATURES)
    price = float(model.predict(row)[0])
    return {"predicted_price": round(price, 2)}


static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def index():
        return FileResponse(static_dir / "index.html")