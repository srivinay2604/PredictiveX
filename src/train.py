import os
import yaml
import pandas as pd
import numpy as np

from src.data.download_data import download_cmapss_data
from src.data.dataset_loader import load_raw_cmapss, calculate_rul, split_by_unit
from src.features.feature_engineering import extract_time_series_features, get_feature_column_names
from src.models.anomaly_detector import AnomalyDetectorAutoencoder
from src.models.failure_classifier import FailureClassifierXGBoost
from src.models.rul_regressor import RULRegressorXGBoost

def run_training_pipeline(config_path: str = "config/config.yaml"):
    print("=" * 60)
    print("STARTING FULL PREDICTIVE MAINTENANCE ML TRAINING PIPELINE")
    print("=" * 60)
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # 1. Download & Load Data
    data_path = download_cmapss_data(config_path)
    df_raw = load_raw_cmapss(data_path)
    
    # 2. Compute Target Labels (RUL, failure_in_horizon)
    max_rul_cap = config["data"].get("max_rul_cap", 125)
    failure_horizon = config["features"].get("failure_horizon_cycles", 30)
    df_labeled = calculate_rul(df_raw, max_rul_cap=max_rul_cap, failure_horizon=failure_horizon)
    
    # 3. Leak-free Unit Split (80% Train Units, 20% Test Units)
    train_raw, test_raw = split_by_unit(df_labeled, test_ratio=0.2, seed=42)
    
    # 4. Feature Engineering
    rolling_windows = config["features"].get("rolling_windows", [5, 10, 20])
    print("\nEngineering Rolling & Degradation Features...")
    train_df = extract_time_series_features(train_raw, rolling_windows=rolling_windows)
    test_df = extract_time_series_features(test_raw, rolling_windows=rolling_windows)
    
    feature_cols = get_feature_column_names(train_df)
    print(f"Total Feature Count: {len(feature_cols)}")
    
    # Save processed data summaries
    train_df.to_parquet("data/processed/train.parquet", index=False)
    test_df.to_parquet("data/processed/test.parquet", index=False)
    print("Saved processed datasets to data/processed/")
    
    # 5. Train PyTorch Autoencoder (Unsupervised Anomaly Detector)
    print("\n--- Training Layer 1: PyTorch Autoencoder Anomaly Detector ---")
    healthy_train = train_df[train_df["RUL"] >= (max_rul_cap * 0.8)].copy()
    
    ae_cfg = config["models"]["autoencoder"]
    anomaly_detector = AnomalyDetectorAutoencoder(
        latent_dim=ae_cfg.get("latent_dim", 8),
        hidden_dims=ae_cfg.get("hidden_dims", [32, 16]),
        percentile_threshold=ae_cfg.get("anomaly_threshold_percentile", 95)
    )
    anomaly_detector.fit(
        healthy_train, 
        feature_cols=feature_cols,
        epochs=ae_cfg.get("epochs", 35),
        batch_size=ae_cfg.get("batch_size", 64),
        lr=ae_cfg.get("learning_rate", 0.001)
    )
    anomaly_detector.save(ae_cfg.get("model_path", "models/autoencoder.pth"))
    
    # Evaluate Anomaly Detector on Test Set
    _, test_anomalies = anomaly_detector.predict(test_df)
    print(f"Test Set Anomaly Flag Ratio: {np.mean(test_anomalies):.2%}")
    
    # 6. Train XGBoost Failure Classifier
    print("\n--- Training Layer 2: XGBoost Failure Classifier ---", flush=True)
    clf_cfg = config["models"]["xgb_classifier"]
    classifier = FailureClassifierXGBoost(
        max_depth=clf_cfg.get("max_depth", 5),
        n_estimators=clf_cfg.get("n_estimators", 50),
        learning_rate=clf_cfg.get("learning_rate", 0.05),
        subsample=clf_cfg.get("subsample", 0.8)
    )
    classifier.fit(train_df, feature_cols=feature_cols, target_col="failure_in_horizon")
    test_clf_metrics = classifier.evaluate(test_df, target_col="failure_in_horizon")
    print("Test Set Classification Performance:", flush=True)
    for k, v in test_clf_metrics.items():
        print(f"  {k}: {v:.4f}", flush=True)
    classifier.save(clf_cfg.get("model_path", "models/xgb_classifier.json"))
    
    # 7. Train XGBoost RUL Regressor
    print("\n--- Training Layer 3: XGBoost RUL Regressor ---", flush=True)
    reg_cfg = config["models"]["xgb_regressor"]
    regressor = RULRegressorXGBoost(
        max_depth=reg_cfg.get("max_depth", 5),
        n_estimators=reg_cfg.get("n_estimators", 50),
        learning_rate=reg_cfg.get("learning_rate", 0.05),
        subsample=reg_cfg.get("subsample", 0.8)
    )
    regressor.fit(train_df, feature_cols=feature_cols, target_col="RUL")
    test_reg_metrics = regressor.evaluate(test_df, target_col="RUL")
    print("Test Set RUL Regression Performance:", flush=True)
    for k, v in test_reg_metrics.items():
        print(f"  {k}: {v:.4f}", flush=True)
    regressor.save(reg_cfg.get("model_path", "models/xgb_regressor.json"))
    
    print("\n=" * 60, flush=True)
    print("ALL MODELS SUCCESSFULLY TRAINED & SAVED TO models/", flush=True)
    print("=" * 60, flush=True)

if __name__ == "__main__":
    run_training_pipeline()
