import os
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict

class RULRegressorXGBoost:
    """
    XGBoost Regression model for continuous Remaining Useful Life (RUL) prediction.
    Uses raw Booster API for inference to avoid segfault on Python 3.14 + Apple Silicon.
    """
    def __init__(self, max_depth: int = 6, n_estimators: int = 200, learning_rate: float = 0.05, subsample: float = 0.8):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.model = None
        self.booster = None  # Raw Booster for safe inference
        self.feature_cols = []

    def fit(self, train_df: pd.DataFrame, feature_cols: list, target_col: str = "RUL_raw") -> Dict[str, float]:
        self.feature_cols = feature_cols
        X = train_df[feature_cols].values
        y = train_df[target_col].values

        self.model = xgb.XGBRegressor(
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=0.8,
            random_state=42,
        )

        self.model.fit(X, y)
        self.booster = self.model.get_booster()

        preds = self.model.predict(X)
        metrics = {
            "train_mae": float(mean_absolute_error(y, preds)),
            "train_rmse": float(np.sqrt(mean_squared_error(y, preds))),
            "train_r2": float(r2_score(y, preds))
        }
        print(f"XGBoost RUL Regressor Trained. MAE: {metrics['train_mae']:.2f}, RMSE: {metrics['train_rmse']:.2f}, R2: {metrics['train_r2']:.4f}")
        return metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise ValueError("XGBoost regressor is not loaded.")
        X = df[self.feature_cols].values.astype(np.float32)
        dmat = xgb.DMatrix(X, feature_names=self.feature_cols)
        return self.booster.predict(dmat)

    def evaluate(self, test_df: pd.DataFrame, target_col: str = "RUL_raw") -> Dict[str, float]:
        y_test = test_df[target_col].values
        preds = self.predict(test_df)
        metrics = {
            "test_mae": float(mean_absolute_error(y_test, preds)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "test_r2": float(r2_score(y_test, preds))
        }
        return metrics

    def save(self, path: str = "models/xgb_regressor.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
        meta_path = path.replace(".json", "_meta.npz")
        np.savez(meta_path, feature_cols=np.array(self.feature_cols))
        print(f"Saved XGBoost RUL Regressor to {path}")

    def load(self, path: str = "models/xgb_regressor.json"):
        # Use raw Booster API to avoid segfault in XGBRegressor.load_model on Python 3.14
        self.booster = xgb.Booster()
        self.booster.load_model(path)
        self.model = None  # sklearn wrapper not available after Booster load
        meta_path = path.replace(".json", "_meta.npz")
        meta = np.load(meta_path)
        self.feature_cols = list(meta["feature_cols"])
        print(f"Loaded XGBoost RUL Regressor from {path}")
        return self
