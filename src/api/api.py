"""
api.py
------
Backend REST API — FastAPI.
Provides /predict, /recommend, /investment, /anomaly, and /health endpoints.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────────────
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.preprocessing.preprocess           import DataPreprocessor
from src.recommendation.recommendation      import PropertyRecommender
from src.investment_analysis.investment_analysis import InvestmentAnalyzer
from src.anomaly_detection.anomaly_detection import AnomalyDetector

MODELS_DIR = ROOT / "models"

# ─────────────────────────────────────────────
app = FastAPI(
    title="🏠 AI Real Estate Analytics API",
    description="Property valuation, recommendations, investment analysis, and anomaly detection.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Lazy-loaded singletons ───────────────────────────────────────────
_preprocessor: Optional[DataPreprocessor]  = None
_best_model                                 = None
_best_model_name: str                       = ""
_recommender: Optional[PropertyRecommender] = None
_analyzer: InvestmentAnalyzer               = InvestmentAnalyzer()
_anomaly_det: Optional[AnomalyDetector]     = None


def _load_prediction_stack():
    global _preprocessor, _best_model, _best_model_name
    if _best_model is None:
        _preprocessor  = DataPreprocessor.load_artifacts()
        model_path = MODELS_DIR / "best_model.pkl"
        with open(model_path, "rb") as f:
            payload = pickle.load(f)
        _best_model      = payload["model"]
        _best_model_name = payload["name"]


def _load_recommender():
    global _recommender
    if _recommender is None:
        _recommender = PropertyRecommender.load()


def _load_anomaly():
    global _anomaly_det
    if _anomaly_det is None:
        _anomaly_det = AnomalyDetector.load()


# ── Request / Response schemas ───────────────────────────────────────
class PropertyInput(BaseModel):
    city:           str             = Field(..., example="Pune")
    locality:       str             = Field(..., example="Baner")
    property_type:  str             = Field(..., example="Apartment")
    bedrooms:       int             = Field(..., ge=1, le=10, example=3)
    bathrooms:      int             = Field(..., ge=1, le=10, example=2)
    area_sqft:      float           = Field(..., gt=0, example=1200)
    age_years:      Optional[float] = Field(None, example=5)
    floor_number:   Optional[int]   = Field(None, example=4)
    total_floors:   Optional[int]   = Field(None, example=12)
    location_score: Optional[float] = Field(None, ge=0, le=10, example=7.5)
    amenities:      Optional[str]   = Field(None, example="Gym|Parking|Security")
    price:          Optional[float] = Field(None, example=8500000)


class PredictionResponse(BaseModel):
    predicted_price:  float
    model_used:       str
    price_range_low:  float
    price_range_high: float
    price_per_sqft:   float


class RecommendRequest(BaseModel):
    property:           PropertyInput
    top_n:              int   = Field(5, ge=1, le=20)
    price_tolerance_pct: float = Field(30.0, ge=0, le=100)
    same_city:          bool  = True


class InvestmentRequest(BaseModel):
    price:     float = Field(..., gt=0, example=8500000)
    city:      str   = Field(..., example="Pune")
    area_sqft: Optional[float] = None
    bedrooms:  Optional[int]   = None


class AnomalyRequest(BaseModel):
    properties: List[PropertyInput]


# ── Endpoints ────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    """Service health check."""
    return {"status": "ok", "service": "Real Estate Analytics API v1.0"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_price(prop: PropertyInput):
    """
    Predict property price using the best trained ML model.

    Returns predicted price, ±10% confidence range, and price/sqft.
    """
    try:
        _load_prediction_stack()
    except Exception as e:
        raise HTTPException(500, f"Model loading failed: {e}. Run train_pipeline.py first.")

    input_df = pd.DataFrame([prop.dict()])
    try:
        X = _preprocessor.transform(input_df)
        price = float(_best_model.predict(X)[0])
    except Exception as e:
        raise HTTPException(422, f"Prediction error: {e}")

    price = max(price, 0)
    ppsf  = round(price / prop.area_sqft, 2) if prop.area_sqft else 0

    return PredictionResponse(
        predicted_price  = round(price, 2),
        model_used       = _best_model_name,
        price_range_low  = round(price * 0.90, 2),
        price_range_high = round(price * 1.10, 2),
        price_per_sqft   = ppsf,
    )


@app.post("/recommend", tags=["Recommendation"])
def recommend_properties(req: RecommendRequest):
    """
    Recommend top-N similar properties based on feature similarity.
    """
    try:
        _load_recommender()
    except Exception as e:
        raise HTTPException(500, f"Recommender loading failed: {e}. Run train_pipeline.py first.")

    query_dict = req.property.dict()
    _recommender.top_n = req.top_n

    try:
        recs = _recommender.recommend(
            query_dict,
            price_tolerance_pct=req.price_tolerance_pct,
            same_city=req.same_city,
        )
    except Exception as e:
        raise HTTPException(422, f"Recommendation error: {e}")

    return {
        "query":           query_dict,
        "recommendations": recs.to_dict(orient="records"),
        "count":           len(recs),
    }


@app.post("/investment", tags=["Investment"])
def investment_analysis(req: InvestmentRequest):
    """
    Calculate ROI, rental yield, EMI, and investment grade for a property.
    """
    report = _analyzer.analyze(
        price     = req.price,
        city      = req.city,
        area_sqft = req.area_sqft,
        bedrooms  = req.bedrooms,
    )
    return report.to_dict()


@app.post("/anomaly", tags=["Anomaly Detection"])
def detect_anomalies(req: AnomalyRequest):
    """
    Flag suspicious / anomalous property listings.
    Accepts a list of property records and returns anomaly scores.
    """
    try:
        _load_anomaly()
    except Exception as e:
        raise HTTPException(500, f"Anomaly detector loading failed: {e}. Run train_pipeline.py first.")

    df = pd.DataFrame([p.dict() for p in req.properties])
    if "price" not in df.columns or df["price"].isnull().all():
        raise HTTPException(422, "Each property must include 'price' for anomaly detection.")

    flagged = _anomaly_det.detect(df)
    cols    = ["anomaly_score", "is_anomaly_iso", "is_anomaly_zscore",
               "is_anomaly_iqr", "anomaly_flag", "anomaly_reason"]
    result  = flagged[[c for c in cols if c in flagged.columns]]

    return {
        "results":         result.to_dict(orient="records"),
        "anomaly_count":   int(flagged["anomaly_flag"].sum()),
        "total_records":   len(flagged),
    }


@app.get("/cities", tags=["Reference"])
def list_cities():
    """Return list of supported cities."""
    return {
        "cities": [
            "Mumbai", "Pune", "Bangalore", "Hyderabad", "Chennai",
            "Delhi", "Kolkata", "Ahmedabad", "Jaipur", "Surat",
        ]
    }


@app.get("/model-info", tags=["System"])
def model_info():
    """Return info about loaded models."""
    try:
        _load_prediction_stack()
        return {
            "best_model": _best_model_name,
            "status":     "loaded",
        }
    except Exception as e:
        return {"best_model": "not loaded", "status": str(e)}


# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.api:app", host="0.0.0.0", port=8000, reload=True)
