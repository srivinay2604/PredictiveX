from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class TelemetrySample(BaseModel):
    unit_id: int = Field(..., example=1, description="Asset / Equipment ID")
    cycle: int = Field(..., example=120, description="Current operational cycle count")
    setting_1: float = Field(default=0.0)
    setting_2: float = Field(default=0.0)
    setting_3: float = Field(default=100.0)
    sensor_2: float = Field(..., example=642.5)
    sensor_3: float = Field(..., example=1588.2)
    sensor_4: float = Field(..., example=1405.1)
    sensor_7: float = Field(..., example=552.1)
    sensor_8: float = Field(..., example=2388.1)
    sensor_9: float = Field(..., example=9055.0)
    sensor_11: float = Field(..., example=47.2)
    sensor_12: float = Field(..., example=520.5)
    sensor_13: float = Field(..., example=2388.0)
    sensor_14: float = Field(..., example=8135.0)
    sensor_15: float = Field(..., example=8.45)
    sensor_17: float = Field(..., example=393.0)
    sensor_20: float = Field(..., example=38.5)
    sensor_21: float = Field(..., example=23.1)

class TelemetryBatchRequest(BaseModel):
    history: List[TelemetrySample]

class FeatureDriver(BaseModel):
    feature: str
    value: float
    shap_value: float
    impact: str

class PredictionResponse(BaseModel):
    unit_id: int
    cycle: int
    anomaly_score: float
    is_anomaly: bool
    failure_probability: float
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    predicted_rul: float
    top_drivers: List[FeatureDriver]
    timestamp: str

class RecommendationRequest(BaseModel):
    unit_id: int
    cycle: int
    failure_probability: float
    predicted_rul: float
    anomaly_score: float
    top_drivers: List[FeatureDriver]

class RecommendationResponse(BaseModel):
    unit_id: int
    urgency: str
    maintenance_summary: str
    action_items: List[str]
    root_cause_explanation: str
