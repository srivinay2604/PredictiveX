import pytest
import pandas as pd
import numpy as np
from src.models.anomaly_detector import AnomalyDetectorAutoencoder
from src.models.failure_classifier import FailureClassifierXGBoost
from src.models.rul_regressor import RULRegressorXGBoost
from src.models.explainer import ModelExplainer

@pytest.fixture
def dummy_dataset():
    np.random.seed(42)
    n = 50
    data = {
        "unit_id": np.random.choice([1, 2], n),
        "cycle": np.tile(np.arange(1, 26), 2),
        "feat_1": np.random.normal(10, 2, n),
        "feat_2": np.random.normal(50, 5, n),
        "RUL": np.random.uniform(1, 100, n),
        "failure_in_horizon": np.random.choice([0, 1], n)
    }
    return pd.DataFrame(data)

def test_autoencoder(dummy_dataset):
    feature_cols = ["feat_1", "feat_2"]
    ae = AnomalyDetectorAutoencoder(latent_dim=4, hidden_dims=[16, 8])
    ae.fit(dummy_dataset, feature_cols=feature_cols, epochs=5, batch_size=16)
    
    scores, flags = ae.predict(dummy_dataset)
    assert len(scores) == len(dummy_dataset)
    assert len(flags) == len(dummy_dataset)
    assert set(np.unique(flags)).issubset({0, 1})

def test_failure_classifier(dummy_dataset):
    feature_cols = ["feat_1", "feat_2"]
    clf = FailureClassifierXGBoost(n_estimators=10, max_depth=3)
    clf.fit(dummy_dataset, feature_cols=feature_cols, target_col="failure_in_horizon")
    
    probas = clf.predict_proba(dummy_dataset)
    assert len(probas) == len(dummy_dataset)
    assert np.all((probas >= 0.0) & (probas <= 1.0))

def test_rul_regressor(dummy_dataset):
    feature_cols = ["feat_1", "feat_2"]
    reg = RULRegressorXGBoost(n_estimators=10, max_depth=3)
    reg.fit(dummy_dataset, feature_cols=feature_cols, target_col="RUL")
    
    preds = reg.predict(dummy_dataset)
    assert len(preds) == len(dummy_dataset)
    assert np.all(preds >= 0.0)

def test_explainer(dummy_dataset):
    feature_cols = ["feat_1", "feat_2"]
    clf = FailureClassifierXGBoost(n_estimators=10, max_depth=3)
    clf.fit(dummy_dataset, feature_cols=feature_cols, target_col="failure_in_horizon")
    
    explainer = ModelExplainer(clf)
    sample = dummy_dataset.iloc[[0]]
    drivers = explainer.explain_sample(sample, top_k=2)
    assert len(drivers) == 2
    assert "feature" in drivers[0]
    assert "shap_value" in drivers[0]
