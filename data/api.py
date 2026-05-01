"""
api.py
------
FastAPI REST API for real-time SMS phishing detection.
Receives SMS messages via HTTP, scores them, and logs results.

Usage:
    uvicorn api:app --reload --port 8000

Endpoints:
    POST /analyze          →  score a single SMS message
    POST /analyze/batch    →  score multiple messages
    GET  /stats            →  detection statistics
    GET  /flagged          →  list of flagged messages
    GET  /health           →  health check
"""

import json
import pickle
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from detector import extract_features, FEATURE_COLS

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SMS Phishing Detection API",
    description="Real-time SMS phishing detection using ML. Inspired by production work at MCI (Iran's largest telecom).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory log (last 1000 results) ────────────────────────────────────────
# In production this would be a database (PostgreSQL / TimescaleDB)
results_log = deque(maxlen=1000)

stats = {
    "total_analyzed": 0,
    "total_phishing": 0,
    "total_normal": 0,
    "high_risk": 0,
    "medium_risk": 0,
    "low_risk": 0,
    "started_at": datetime.utcnow().isoformat(),
}

# ── Load model on startup ─────────────────────────────────────────────────────

MODEL_PATH = "../../models/phishing_model.pkl"
pipeline = None

@app.on_event("startup")
def load_model():
    global pipeline
    try:
        with open(MODEL_PATH, "rb") as f:
            pipeline = pickle.load(f)
        print(f"Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"WARNING: Model not found at {MODEL_PATH}. Run train_model.py first.")


# ── Request / Response schemas ────────────────────────────────────────────────

class SMSRequest(BaseModel):
    message: str
    sender: Optional[str] = "UNKNOWN"
    timestamp: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "message": "URGENT: Your account is suspended. Verify now: http://mci-secure.xyz/abc",
                "sender": "INFO",
                "timestamp": "2026-05-01T10:00:00"
            }
        }

class BatchRequest(BaseModel):
    messages: List[SMSRequest]

class DetectionResult(BaseModel):
    id: str
    timestamp: str
    sender: str
    message_preview: str
    prediction: str
    confidence: float
    risk_level: str
    signals: List[str]


# ── Helper ────────────────────────────────────────────────────────────────────

def analyze_sms(sms: SMSRequest) -> DetectionResult:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py first.")

    features = extract_features(sms.message)
    X = pd.DataFrame([features])[FEATURE_COLS]

    prediction = int(pipeline.predict(X)[0])
    confidence = round(float(pipeline.predict_proba(X)[0][1]), 4)

    if confidence >= 0.85:
        risk_level = "HIGH"
    elif confidence >= 0.50:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Build signals list
    signals = []
    if features["has_url"]:
        signals.append("Contains URL")
    if features["has_suspicious_tld"]:
        signals.append("Suspicious domain extension (.xyz, .tk, .ru, etc.)")
    if features["has_urgency_words"]:
        signals.append("Urgency language detected")
    if features["has_prize_words"]:
        signals.append("Prize or reward language detected")
    if features["url_shortener_present"]:
        signals.append("URL shortener detected")

    result = DetectionResult(
        id=str(uuid.uuid4()),
        timestamp=sms.timestamp or datetime.utcnow().isoformat(),
        sender=sms.sender or "UNKNOWN",
        message_preview=sms.message[:100] + ("..." if len(sms.message) > 100 else ""),
        prediction="PHISHING" if prediction == 1 else "NORMAL",
        confidence=confidence,
        risk_level=risk_level,
        signals=signals,
    )

    # Update stats
    stats["total_analyzed"] += 1
    if prediction == 1:
        stats["total_phishing"] += 1
    else:
        stats["total_normal"] += 1
    stats[f"{risk_level.lower()}_risk"] += 1

    # Log result
    results_log.appendleft(result.dict())

    return result


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/analyze", response_model=DetectionResult)
def analyze_single(sms: SMSRequest):
    """Score a single SMS message and return phishing verdict."""
    return analyze_sms(sms)


@app.post("/analyze/batch", response_model=List[DetectionResult])
def analyze_batch(batch: BatchRequest):
    """Score multiple SMS messages in one request."""
    if len(batch.messages) > 500:
        raise HTTPException(status_code=400, detail="Maximum 500 messages per batch request.")
    return [analyze_sms(sms) for sms in batch.messages]


@app.get("/stats")
def get_stats():
    """Return detection statistics since server started."""
    total = stats["total_analyzed"]
    return {
        **stats,
        "phishing_rate": round(stats["total_phishing"] / total, 4) if total > 0 else 0,
    }


@app.get("/flagged")
def get_flagged(limit: int = 50, risk_level: Optional[str] = None):
    """Return recently flagged phishing messages."""
    flagged = [r for r in results_log if r["prediction"] == "PHISHING"]
    if risk_level:
        flagged = [r for r in flagged if r["risk_level"] == risk_level.upper()]
    return {
        "count": len(flagged[:limit]),
        "results": flagged[:limit],
    }


@app.get("/")
def root():
    return {
        "service": "SMS Phishing Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": ["/analyze", "/analyze/batch", "/stats", "/flagged"],
    }