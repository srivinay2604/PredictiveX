import sys
import os
import pandas as pd
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_features import test_fft_features, test_vibration_features, test_extract_time_series_features
from tests.test_models import dummy_dataset, test_autoencoder, test_failure_classifier, test_rul_regressor, test_explainer

def run_tests():
    print("============================================================")
    print("RUNNING END-TO-END SYSTEM INTEGRATION & UNIT TESTS")
    print("============================================================")
    
    print("\n1. Testing FFT Feature Processing...", end=" ")
    test_fft_features()
    print("PASSED ✅")
    
    print("2. Testing Time-Domain Vibration Metrics...", end=" ")
    test_vibration_features()
    print("PASSED ✅")
    
    print("3. Testing Rolling Feature Engineering Pipeline...", end=" ")
    test_extract_time_series_features()
    print("PASSED ✅")
    
    dataset = dummy_dataset()
    
    print("4. Testing PyTorch Autoencoder Anomaly Detector...", end=" ")
    test_autoencoder(dataset)
    print("PASSED ✅")
    
    print("5. Testing Supervised XGBoost Failure Classifier...", end=" ")
    test_failure_classifier(dataset)
    print("PASSED ✅")
    
    print("6. Testing Continuous XGBoost RUL Regressor...", end=" ")
    test_rul_regressor(dataset)
    print("PASSED ✅")
    
    print("7. Testing SHAP Feature Attributer & Explainer...", end=" ")
    test_explainer(dataset)
    print("PASSED ✅")
    
    print("\n============================================================")
    print("ALL 7 SYSTEM & ML MODULE TESTS PASSED PERFECTLY! 🎉")
    print("============================================================")

if __name__ == "__main__":
    run_tests()
