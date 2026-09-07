import pytest
import pandas as pd
import numpy as np
from src.features.signal_processing import compute_fft_features, compute_time_domain_vibration_features
from src.features.feature_engineering import extract_time_series_features, get_feature_column_names

def test_fft_features():
    signal = np.sin(np.linspace(0, 10 * np.pi, 100)) + np.random.normal(0, 0.1, 100)
    features = compute_fft_features(signal, sample_rate=100.0)
    assert "spectral_centroid" in features
    assert "spectral_energy" in features
    assert "dominant_frequency" in features
    assert features["spectral_energy"] > 0

def test_vibration_features():
    signal = np.random.normal(0, 1.0, 100)
    features = compute_time_domain_vibration_features(signal)
    assert "rms" in features
    assert "crest_factor" in features
    assert "kurtosis" in features
    assert "peak_to_peak" in features

def test_extract_time_series_features():
    data = {
        "unit_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "cycle": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "sensor_2": np.random.uniform(600, 650, 10),
        "sensor_3": np.random.uniform(1500, 1600, 10),
        "sensor_7": np.random.uniform(500, 550, 10),
        "RUL": [100, 99, 98, 97, 96, 120, 119, 118, 117, 116],
        "failure_in_horizon": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    df = pd.DataFrame(data)
    engineered = extract_time_series_features(df, rolling_windows=[3])
    feature_cols = get_feature_column_names(engineered)
    
    assert len(engineered) == 10
    assert len(feature_cols) > 3
    assert not engineered[feature_cols].isnull().any().any()
