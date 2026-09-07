import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

class TelemetrySimulator:
    """
    SCADA/IIoT Telemetry Playback Simulator.
    Replays sequential sensor streams for an asset run-to-failure lifecycle.
    """
    def __init__(self, data_path: str = "data/raw/train_FD001.txt"):
        from src.data.dataset_loader import load_raw_cmapss, calculate_rul
        if not os.path.exists(data_path):
            from src.data.download_data import download_cmapss_data
            data_path = download_cmapss_data()
        raw_df = load_raw_cmapss(data_path)
        self.df = calculate_rul(raw_df)
        self.units = sorted(self.df["unit_id"].unique())
        
    def get_unit_max_cycle(self, unit_id: int) -> int:
        unit_data = self.df[self.df["unit_id"] == unit_id]
        return int(unit_data["cycle"].max())

    def get_telemetry_history(self, unit_id: int, current_cycle: int, window_size: int = 20) -> pd.DataFrame:
        """
        Retrieves historical sensor telemetry leading up to current_cycle for feature engineering calculation.
        """
        unit_data = self.df[self.df["unit_id"] == unit_id].sort_values(by="cycle")
        start_cycle = max(1, current_cycle - window_size + 1)
        sub_df = unit_data[(unit_data["cycle"] >= start_cycle) & (unit_data["cycle"] <= current_cycle)].copy()
        return sub_df

    def get_fleet_status_snapshot(self, target_cycle_ratio: float = 0.75) -> List[Dict[str, Any]]:
        """
        Returns a fleet-wide status snapshot where each unit is simulated at a specific point in its lifecycle.
        """
        snapshot = []
        for u in self.units[:30]:  # Top 30 units for fleet dashboard
            max_c = self.get_unit_max_cycle(u)
            c = int(max_c * target_cycle_ratio)
            c = max(1, min(c, max_c))
            row = self.df[(self.df["unit_id"] == u) & (self.df["cycle"] == c)].iloc[0]
            snapshot.append({
                "unit_id": int(u),
                "current_cycle": int(c),
                "max_cycle": int(max_c),
                "RUL_raw": int(row["RUL_raw"]),
                "sensor_2": float(row["sensor_2"]),
                "sensor_3": float(row["sensor_3"]),
                "sensor_4": float(row["sensor_4"]),
                "sensor_7": float(row["sensor_7"])
            })
        return snapshot
