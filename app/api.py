"""
app/api.py
Home Credit Default Risk — FastAPI prediction service.

Endpoints:
    GET  /health            — liveness check
    GET  /models            — list loaded models + metadata
    POST /predict           — single applicant prediction (both models)
    POST /predict/batch     — batch predictions (list of applicants)

Usage:
    uvicorn app.api:app --reload --port 8000
    # then: curl http://localhost:8000/health
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.train import CreditRiskNN, MODELS_DIR, DEVICE
from src.features import build_features, EXT_SOURCES
from src.data import PROC_DIR

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Risk Scorer",
    description="PyTorch NN + XGBoost credit default probability API",
    version="1.0.0",
)

# ── startup: load models once ─────────────────────────────────────────────────

_state: dict[str, Any] = {}


def _load_models() -> None:
    """Load both models and scaler into module-level state."""
    # XGBoost
    xgb_path = MODELS_DIR / "xgb_baseline.joblib"
    if not xgb_path.exists():
        raise RuntimeError(f"XGBoost model not found at {xgb_path}. Run src.train first.")
    _state["xgb"] = joblib.load(xgb_path)
    logger.info("XGBoost loaded from %s", xgb_path)

    # NN
    pt_path = MODELS_DIR / "nn_credit_risk.pt"
    sc_path = MODELS_DIR / "nn_scaler.joblib"
    if not pt_path.exists():
        raise RuntimeError(f"NN model not found at {pt_path}. Run src.train first.")
    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=True)
    n_features = checkpoint["n_features"]
    nn_model = CreditRiskNN(n_features).to(DEVICE)
    nn_model.load_state_dict(checkpoint["model_state"])
    nn_model.eval()
    _state["nn"]       = nn_model
    _state["scaler"]   = joblib.load(sc_path)
    _state["n_features"] = n_features
    logger.info("NN loaded (%d features) from %s", n_features, pt_path)

    # Feature names
    feat_path = PROC_DIR / "feature_names_eng.txt"
    if feat_path.exists():
        _state["feature_names"] = feat_path.read_text().splitlines()
    logger.info("Models ready.")


@app.on_event("startup")
def startup_event() -> None:
    logging.basicConfig(level=logging.INFO)
    _load_models()


# ── schemas ───────────────────────────────────────────────────────────────────

class ApplicantFeatures(BaseModel):
    """
    Raw applicant features — matches Home Credit application columns.
    Only the most predictive fields are required; others default to 0.
    All monetary amounts in local currency units.
    """
    # External credit scores (most predictive)
    EXT_SOURCE_1: float = Field(0.5, ge=0.0, le=1.0, description="External credit score 1 (0–1)")
    EXT_SOURCE_2: float = Field(0.5, ge=0.0, le=1.0, description="External credit score 2 (0–1)")
    EXT_SOURCE_3: float = Field(0.5, ge=0.0, le=1.0, description="External credit score 3 (0–1)")

    # Loan amounts
    AMT_CREDIT:      float = Field(500_000, gt=0, description="Total credit amount")
    AMT_ANNUITY:     float = Field(25_000,  gt=0, description="Loan annuity (monthly payment × 12)")
    AMT_INCOME_TOTAL:float = Field(180_000, gt=0, description="Total annual income")
    AMT_GOODS_PRICE: float = Field(450_000, gt=0, description="Price of goods for loan")

    # Demographics
    DAYS_BIRTH:     float = Field(-12000, lt=0, description="Days before application (negative)")
    DAYS_EMPLOYED:  float = Field(-2000,  description="Days employed before application (negative; 365243=unemployed)")
    CNT_CHILDREN:   float = Field(0, ge=0)
    CNT_FAM_MEMBERS:float = Field(2, ge=1)

    # Categorical codes (label-encoded; 0=most common category)
    NAME_CONTRACT_TYPE:  float = Field(0)
    CODE_GENDER:         float = Field(0)
    FLAG_OWN_CAR:        float = Field(0)
    FLAG_OWN_REALTY:     float = Field(0)

    class Config:
        json_schema_extra = {
            "example": {
                "EXT_SOURCE_1": 0.60,
                "EXT_SOURCE_2": 0.55,
                "EXT_SOURCE_3": 0.45,
                "AMT_CREDIT": 450000,
                "AMT_ANNUITY": 22500,
                "AMT_INCOME_TOTAL": 150000,
                "AMT_GOODS_PRICE": 400000,
                "DAYS_BIRTH": -14600,
                "DAYS_EMPLOYED": -3650,
                "CNT_CHILDREN": 1,
                "CNT_FAM_MEMBERS": 3,
            }
        }


class PredictionResponse(BaseModel):
    xgb_default_prob:  float = Field(..., description="XGBoost default probability (0–1)")
    nn_default_prob:   float = Field(..., description="PyTorch NN default probability (0–1)")
    xgb_risk_label:    str   = Field(..., description="LOW / MEDIUM / HIGH")
    nn_risk_label:     str   = Field(..., description="LOW / MEDIUM / HIGH")
    inference_ms:      float = Field(..., description="Inference time in milliseconds")


class BatchRequest(BaseModel):
    applicants: list[ApplicantFeatures]


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    total_ms:    float


# ── helpers ───────────────────────────────────────────────────────────────────

def _risk_label(prob: float) -> str:
    if prob < 0.10:
        return "LOW"
    if prob < 0.25:
        return "MEDIUM"
    return "HIGH"


def _applicant_to_array(applicant: ApplicantFeatures, feature_names: list[str]) -> np.ndarray:
    """
    Convert an ApplicantFeatures instance to a 1-row float32 array
    aligned to the full 143-feature engineered schema.
    Unspecified features default to 0.
    """
    row = {f: 0.0 for f in feature_names}

    # fill in provided fields
    provided = applicant.model_dump()
    for col, val in provided.items():
        if col in row:
            row[col] = float(val)

    X_base = np.array([[row[f] for f in feature_names[:120]]], dtype=np.float32)
    # use a zero placeholder for val; build_features is stateless
    _, X_eng, _ = build_features(X_base, X_base, feature_names[:120])
    return X_eng  # shape (1, 143)


def _predict_row(X: np.ndarray) -> tuple[float, float]:
    """Return (xgb_prob, nn_prob) for a single engineered row."""
    xgb_prob = float(_state["xgb"].predict_proba(X)[:, 1][0])

    X_s = _state["scaler"].transform(X).astype(np.float32)
    with torch.no_grad():
        tensor = torch.tensor(X_s, dtype=torch.float32).to(DEVICE)
        logit  = _state["nn"](tensor).squeeze(1)
        nn_prob = float(logit.sigmoid().item())

    return xgb_prob, nn_prob


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "models_loaded": list(_state.keys()),
        "device": str(DEVICE),
    }


@app.get("/models")
def models_info() -> dict:
    xgb = _state.get("xgb")
    return {
        "xgb": {
            "type": "XGBClassifier",
            "best_iteration": getattr(xgb, "best_iteration", None),
            "n_features": _state.get("n_features"),
        },
        "nn": {
            "type": "CreditRiskNN (attention MLP)",
            "n_features": _state.get("n_features"),
            "device": str(DEVICE),
        },
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(applicant: ApplicantFeatures) -> PredictionResponse:
    feat_names = _state.get("feature_names")
    if not feat_names:
        raise HTTPException(status_code=503, detail="Feature names not loaded.")

    t0 = time.perf_counter()
    try:
        X = _applicant_to_array(applicant, feat_names)
        xgb_prob, nn_prob = _predict_row(X)
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(e))

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return PredictionResponse(
        xgb_default_prob = round(xgb_prob, 6),
        nn_default_prob  = round(nn_prob,  6),
        xgb_risk_label   = _risk_label(xgb_prob),
        nn_risk_label    = _risk_label(nn_prob),
        inference_ms     = round(elapsed_ms, 2),
    )


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest) -> BatchResponse:
    if len(batch.applicants) > 500:
        raise HTTPException(status_code=400, detail="Batch size limit is 500.")

    feat_names = _state.get("feature_names")
    if not feat_names:
        raise HTTPException(status_code=503, detail="Feature names not loaded.")

    t0 = time.perf_counter()
    predictions = []

    for applicant in batch.applicants:
        t1 = time.perf_counter()
        X  = _applicant_to_array(applicant, feat_names)
        xgb_prob, nn_prob = _predict_row(X)
        row_ms = (time.perf_counter() - t1) * 1000
        predictions.append(PredictionResponse(
            xgb_default_prob = round(xgb_prob, 6),
            nn_default_prob  = round(nn_prob,  6),
            xgb_risk_label   = _risk_label(xgb_prob),
            nn_risk_label    = _risk_label(nn_prob),
            inference_ms     = round(row_ms, 2),
        ))

    total_ms = (time.perf_counter() - t0) * 1000
    return BatchResponse(predictions=predictions, total_ms=round(total_ms, 2))
