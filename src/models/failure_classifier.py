import os
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, accuracy_score
from typing import Dict, Any

class FailureClassifierXGBoost:
    """
    Supervised Gradient Boosted Classifier (XGBoost) for failure prediction within a defined future horizon.
    Uses raw Booster API for inference to avoid segfault on Python 3.14 + Apple Silicon.
    """
    def __init__(self, max_depth: int = 6, n_estimators: int = 150, learning_rate: float = 0.05, subsample: float = 0.8):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.model = None
        self.booster = None  # Raw Booster for safe inference
        self.feature_cols = []
        
    def fit(self, train_df: pd.DataFrame, feature_cols: list, target_col: str = "failure_in_horizon") -> Dict[str, float]:
        self.feature_cols = feature_cols
        X = train_df[feature_cols].values
        y = train_df[target_col].values
        
        # Calculate scale_pos_weight for handling class imbalance
        neg_count = np.sum(y == 0)
        pos_count = np.sum(y == 1)
        scale_pos_weight = neg_count / max(1, pos_count)
        
        self.model = xgb.XGBClassifier(
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss"
        )
        
        self.model.fit(X, y)
        self.booster = self.model.get_booster()
        
        preds_proba = self.model.predict_proba(X)[:, 1]
        preds_binary = (preds_proba >= 0.5).astype(int)
        
        metrics = {
            "train_roc_auc": float(roc_auc_score(y, preds_proba)),
            "train_pr_auc": float(average_precision_score(y, preds_proba)),
            "train_f1": float(f1_score(y, preds_binary)),
            "train_precision": float(precision_score(y, preds_binary)),
            "train_recall": float(recall_score(y, preds_binary)),
            "train_accuracy": float(accuracy_score(y, preds_binary))
        }
        print(f"XGBoost Classifier Trained. ROC-AUC: {metrics['train_roc_auc']:.4f}, PR-AUC: {metrics['train_pr_auc']:.4f}, F1: {metrics['train_f1']:.4f}")
        return metrics

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Returns probability of failure (class 1) for each sample."""
        if self.booster is None:
            raise ValueError("XGBoost classifier is not loaded.")
        X = df[self.feature_cols].values.astype(np.float32)
        dmat = xgb.DMatrix(X, feature_names=self.feature_cols)
        # Booster returns probabilities directly for binary:logistic objective
        raw_preds = self.booster.predict(dmat)
        return raw_preds

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probas = self.predict_proba(df)
        return (probas >= threshold).astype(int)

    def evaluate(self, test_df: pd.DataFrame, target_col: str = "failure_in_horizon") -> Dict[str, float]:
        y_test = test_df[target_col].values
        
        probas = self.predict_proba(test_df)
        preds = (probas >= 0.5).astype(int)
        
        metrics = {
            "test_roc_auc": float(roc_auc_score(y_test, probas)),
            "test_pr_auc": float(average_precision_score(y_test, probas)),
            "test_f1": float(f1_score(y_test, preds)),
            "test_precision": float(precision_score(y_test, preds)),
            "test_recall": float(recall_score(y_test, preds)),
            "test_accuracy": float(accuracy_score(y_test, preds))
        }
        return metrics

    def save(self, path: str = "models/xgb_classifier.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
        meta_path = path.replace(".json", "_meta.npz")
        np.savez(meta_path, feature_cols=np.array(self.feature_cols))
        print(f"Saved XGBoost Classifier to {path}")

    def load(self, path: str = "models/xgb_classifier.json"):
        # Use raw Booster API to avoid segfault in XGBClassifier.load_model on Python 3.14
        self.booster = xgb.Booster()
        self.booster.load_model(path)
        self.model = None  # sklearn wrapper not available after Booster load
        meta_path = path.replace(".json", "_meta.npz")
        meta = np.load(meta_path)
        self.feature_cols = list(meta["feature_cols"])
        print(f"Loaded XGBoost Classifier from {path}")
        return self
