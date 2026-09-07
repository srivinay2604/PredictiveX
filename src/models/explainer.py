import shap
import numpy as np
import pandas as pd
from typing import Dict, List, Any

class ModelExplainer:
    """
    SHAP-based Explainability engine for XGBoost models.
    Identifies top physical sensor drivers for asset risk scores.
    Compatible with raw Booster API (no sklearn wrapper needed).
    """
    def __init__(self, model):
        # Extract the raw booster or model for SHAP
        if hasattr(model, "booster") and model.booster is not None:
            # Our custom wrapper with Booster API
            self.underlying = model.booster
            self.feature_cols = model.feature_cols
        elif hasattr(model, "model") and model.model is not None:
            # sklearn-wrapped XGBoost model
            self.underlying = model.model
            self.feature_cols = model.feature_cols
        else:
            self.underlying = model
            self.feature_cols = getattr(model, "feature_cols", [])
            
        self.explainer = shap.TreeExplainer(self.underlying)

    def explain_sample(self, df_sample: pd.DataFrame, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Calculates SHAP feature importances for a single telemetry sample row.
        Returns a sorted list of top-K feature drivers with feature name, reading, and SHAP value.
        """
        if len(df_sample) == 0:
            return []
            
        X = df_sample[self.feature_cols].iloc[[0]]
        shap_values = self.explainer.shap_values(X)
        
        # Handle binary classifier (where shap_values might be 2D array or 1D)
        if isinstance(shap_values, list):
            vals = shap_values[1][0] # Positive class SHAP values
        elif len(shap_values.shape) == 2:
            vals = shap_values[0]
        else:
            vals = shap_values
            
        drivers = []
        for name, val, raw_val in zip(self.feature_cols, vals, X.values[0]):
            drivers.append({
                "feature": name,
                "value": float(raw_val),
                "shap_value": float(val),
                "impact": "increases_risk" if val > 0 else "decreases_risk"
            })
            
        # Sort by absolute SHAP impact magnitude
        drivers = sorted(drivers, key=lambda x: abs(x["shap_value"]), reverse=True)
        return drivers[:top_k]

    def get_summary_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes mean absolute SHAP value across a batch dataset.
        """
        X = df[self.feature_cols]
        shap_values = self.explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            vals = np.abs(shap_values[1]).mean(axis=0)
        else:
            vals = np.abs(shap_values).mean(axis=0)
            
        imp_df = pd.DataFrame({
            "feature": self.feature_cols,
            "mean_abs_shap": vals
        }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
        
        return imp_df
