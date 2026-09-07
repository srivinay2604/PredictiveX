import os
import yaml
from datetime import datetime
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    TelemetryBatchRequest, PredictionResponse, RecommendationRequest, 
    RecommendationResponse, FeatureDriver
)
from src.features.feature_engineering import extract_time_series_features, get_feature_column_names
from src.models.anomaly_detector import AnomalyDetectorAutoencoder
from src.models.failure_classifier import FailureClassifierXGBoost
from src.models.rul_regressor import RULRegressorXGBoost
from src.models.explainer import ModelExplainer
from src.llm.reporter import MaintenanceReportGenerator
from src.api.simulator import TelemetrySimulator

app = FastAPI(
    title="Predictive Maintenance & Asset Health Intelligence API",
    description="Production-grade AI/ML API for Industrial Asset Health Monitoring, Anomaly Detection, XGBoost Failure Prediction, RUL Estimation, and SHAP Root-Cause Attribution.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
models_loaded = False
anomaly_detector = None
classifier = None
regressor = None
explainer = None
reporter = None
simulator = None

def load_all_models():
    global models_loaded, anomaly_detector, classifier, regressor, explainer, reporter, simulator
    
    if os.path.exists("models/autoencoder.pth") and os.path.exists("models/xgb_classifier.json"):
        try:
            anomaly_detector = AnomalyDetectorAutoencoder().load("models/autoencoder.pth")
            classifier = FailureClassifierXGBoost().load("models/xgb_classifier.json")
            regressor = RULRegressorXGBoost().load("models/xgb_regressor.json")
            explainer = ModelExplainer(classifier)
            reporter = MaintenanceReportGenerator()
            simulator = TelemetrySimulator()
            models_loaded = True
            print("Successfully loaded all models and services into FastAPI memory!")
        except Exception as e:
            print(f"Error loading models: {e}")
            models_loaded = False

@app.on_event("startup")
def startup_event():
    load_all_models()

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": models_loaded
    }

@app.post("/predict/telemetry", response_model=PredictionResponse)
def predict_telemetry(request: TelemetryBatchRequest):
    if not models_loaded:
        load_all_models()
        if not models_loaded:
            raise HTTPException(status_code=500, detail="ML Models are not trained or loaded. Run training pipeline first.")

    raw_dicts = [sample.dict() for sample in request.history]
    df_raw = pd.DataFrame(raw_dicts)
    
    if df_raw.empty:
        raise HTTPException(status_code=400, detail="History cannot be empty.")
        
    # Engineer rolling features on sample sequence
    df_feat = extract_time_series_features(df_raw)
    latest_row = df_feat.iloc[[-1]].copy()
    
    unit_id = int(latest_row["unit_id"].values[0])
    cycle = int(latest_row["cycle"].values[0])
    
    # 1. Anomaly Detection (PyTorch Autoencoder)
    mse_scores, is_anom_flags = anomaly_detector.predict(latest_row)
    anomaly_score = float(mse_scores[0])
    is_anomaly = bool(is_anom_flags[0])
    
    # 2. Failure Classification (XGBoost)
    fail_prob = float(classifier.predict_proba(latest_row)[0])
    
    # Risk Level Determination
    if fail_prob >= 0.80:
        risk_level = "CRITICAL"
    elif fail_prob >= 0.50:
        risk_level = "HIGH"
    elif fail_prob >= 0.25:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
        
    # 3. RUL Regression (XGBoost)
    predicted_rul = float(regressor.predict(latest_row)[0])
    
    # 4. SHAP Feature Attribution
    shap_drivers_raw = explainer.explain_sample(latest_row, top_k=5)
    top_drivers = [FeatureDriver(**d) for d in shap_drivers_raw]
    
    return PredictionResponse(
        unit_id=unit_id,
        cycle=cycle,
        anomaly_score=anomaly_score,
        is_anomaly=is_anomaly,
        failure_probability=fail_prob,
        risk_level=risk_level,
        predicted_rul=predicted_rul,
        top_drivers=top_drivers,
        timestamp=datetime.now().isoformat()
    )

@app.post("/explain/recommendation", response_model=RecommendationResponse)
def get_recommendation(req: RecommendationRequest):
    if reporter is None:
        rep = MaintenanceReportGenerator()
    else:
        rep = reporter
        
    drivers_dict = [d.dict() for d in req.top_drivers]
    report = rep.generate_report(
        unit_id=req.unit_id,
        cycle=req.cycle,
        failure_prob=req.failure_probability,
        predicted_rul=req.predicted_rul,
        anomaly_score=req.anomaly_score,
        top_drivers=drivers_dict
    )
    return RecommendationResponse(**report)

@app.get("/fleet/snapshot")
def get_fleet_snapshot(cycle_ratio: float = 0.75):
    if simulator is None:
        sim = TelemetrySimulator()
    else:
        sim = simulator
    return sim.get_fleet_status_snapshot(target_cycle_ratio=cycle_ratio)
