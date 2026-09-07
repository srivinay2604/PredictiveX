import os
import pandas as pd
import numpy as np
import yaml
from typing import Tuple

def load_raw_cmapss(data_path: str = "data/raw/train_FD001.txt") -> pd.DataFrame:
    """
    Loads raw NASA C-MAPSS sensor telemetry data.
    """
    if not os.path.exists(data_path):
        from src.data.download_data import download_cmapss_data
        data_path = download_cmapss_data()

    cols = ["unit_id", "cycle", "setting_1", "setting_2", "setting_3"] + [f"sensor_{i}" for i in range(1, 22)]
    # Handle potential trailing whitespace/empty columns in C-MAPSS text files
    df = pd.read_csv(data_path, sep=r"\s+", header=None, names=cols, index_col=False)
    # Drop sensors with zero variance (static values in C-MAPSS FD001: 1, 5, 10, 16, 18, 19)
    constant_sensors = ["sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]
    df = df.drop(columns=[c for c in constant_sensors if c in df.columns])
    return df

def calculate_rul(df: pd.DataFrame, max_rul_cap: int = 125, failure_horizon: int = 30) -> pd.DataFrame:
    """
    Calculates Remaining Useful Life (RUL) and failure_in_horizon binary classification target.
    """
    df = df.copy()
    max_cycles = df.groupby("unit_id")["cycle"].transform("max")
    raw_rul = max_cycles - df["cycle"]
    
    # Target 1: RUL clipped at max_rul_cap for regression standard benchmark
    df["RUL"] = raw_rul.clip(upper=max_rul_cap)
    df["RUL_raw"] = raw_rul
    
    # Target 2: Binary classification target (will fail within failure_horizon cycles)
    df["failure_in_horizon"] = (raw_rul <= failure_horizon).astype(int)
    
    return df

def split_by_unit(df: pd.DataFrame, test_ratio: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits units cleanly into train and test sets by unit_id to prevent temporal data leakage across time windows.
    """
    units = df["unit_id"].unique()
    np.random.seed(seed)
    np.random.shuffle(units)
    
    num_test = int(len(units) * test_ratio)
    test_units = set(units[:num_test])
    train_units = set(units[num_test:])
    
    train_df = df[df["unit_id"].isin(train_units)].copy().reset_index(drop=True)
    test_df = df[df["unit_id"].isin(test_units)].copy().reset_index(drop=True)
    
    print(f"Group Split Complete: {len(train_units)} Train Units ({len(train_df)} rows), {len(test_units)} Test Units ({len(test_df)} rows)")
    return train_df, test_df

if __name__ == "__main__":
    df_raw = load_raw_cmapss()
    df_labeled = calculate_rul(df_raw)
    train_df, test_df = split_by_unit(df_labeled)
    print("Sample Labeled Head:")
    print(train_df[["unit_id", "cycle", "sensor_2", "RUL", "failure_in_horizon"]].head())
