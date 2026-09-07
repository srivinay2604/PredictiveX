import pandas as pd
import numpy as np
from typing import List

def extract_time_series_features(
    df: pd.DataFrame, 
    sensor_cols: List[str] = None, 
    rolling_windows: List[int] = [5, 10, 20]
) -> pd.DataFrame:
    """
    Engineers rolling statistics, degradation trends, and interaction ratios
    partitioned strictly by unit_id to avoid cross-unit leakage.
    """
    df = df.copy()
    if sensor_cols is None:
        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
        
    df = df.sort_values(by=["unit_id", "cycle"]).reset_index(drop=True)
    
    feature_dfs = [df]
    
    # 1. Rolling statistics per sensor per unit
    for window in rolling_windows:
        rolling_mean = df.groupby("unit_id")[sensor_cols].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        rolling_mean.columns = [f"{col}_roll_mean_{window}" for col in sensor_cols]
        
        rolling_std = df.groupby("unit_id")[sensor_cols].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        ).fillna(0.0)
        rolling_std.columns = [f"{col}_roll_std_{window}" for col in sensor_cols]
        
        feature_dfs.extend([rolling_mean, rolling_std])
        
    # 2. Baseline Deviation Features (difference from cycle 1 healthy operating state)
    first_cycles = df.groupby("unit_id")[sensor_cols].transform("first")
    baseline_diff = (df[sensor_cols] - first_cycles)
    baseline_diff.columns = [f"{col}_dev_baseline" for col in sensor_cols]
    feature_dfs.append(baseline_diff)
    
    # 3. Domain Sensor Interactions (Physical process ratios: Thermal & Pressure ratios)
    if "sensor_2" in df.columns and "sensor_7" in df.columns:
        ratio_df = pd.DataFrame()
        ratio_df["ratio_temp_press_2_7"] = df["sensor_2"] / (df["sensor_7"] + 1e-5)
        if "sensor_3" in df.columns and "sensor_4" in df.columns:
            ratio_df["ratio_temp_3_4"] = df["sensor_3"] / (df["sensor_4"] + 1e-5)
        feature_dfs.append(ratio_df)
        
    final_df = pd.concat(feature_dfs, axis=1)
    # Fill remaining NaNs if any
    final_df = final_df.fillna(0.0)
    
    return final_df

def get_feature_column_names(df: pd.DataFrame) -> List[str]:
    """
    Returns all engineered feature column names excluding metadata and target labels.
    """
    exclude_cols = {"unit_id", "cycle", "RUL", "RUL_raw", "failure_in_horizon", "setting_1", "setting_2", "setting_3"}
    return [c for c in df.columns if c not in exclude_cols]

if __name__ == "__main__":
    from src.data.dataset_loader import load_raw_cmapss, calculate_rul
    raw = load_raw_cmapss()
    labeled = calculate_rul(raw)
    engineered = extract_time_series_features(labeled)
    features = get_feature_column_names(engineered)
    print(f"Original shape: {labeled.shape}, Engineered shape: {engineered.shape}")
    print(f"Total engineered features: {len(features)}")
