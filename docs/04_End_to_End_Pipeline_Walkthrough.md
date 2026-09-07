# 🔗 End-to-End Pipeline Walkthrough

This document traces the **exact data flow** from raw sensor readings to the final maintenance recommendation displayed on the dashboard — every transformation, every model, every output.

---

## The Complete Data Journey

```
[Raw Sensor] → [Clean] → [Engineer Features] → [3 ML Models] → [SHAP] → [NLP Report] → [Dashboard]
```

---

## Step 1: Raw Data Acquisition

**File**: `src/data/download_data.py`

### What happens:
1. The system checks if `data/raw/train_FD001.txt` exists
2. If not, it generates **synthetic turbofan sensor data** simulating 100 engine units
3. Each unit runs from cycle 1 to ~200 cycles (normally distributed)
4. 21 sensors are simulated with:
   - **Physics-based degradation**: `degradation = (cycle / max_cycle)²` — quadratic wear curve
   - **Sensor drift**: Key sensors (temp, pressure, speed) shift linearly with degradation
   - **Gaussian noise**: Added at realistic amplitudes per sensor type

### Example raw row:
```
unit_id  cycle  setting_1  setting_2  setting_3  sensor_1  sensor_2  ...  sensor_21
1        1      0.0        0.0        100.0      518.67    642.01    ...  23.30
1        2      0.0        0.0        100.0      518.67    642.12    ...  23.29
...
1        192    0.0        0.0        100.0      518.67    656.84    ...  22.10
```

---

## Step 2: Data Loading & Cleaning

**File**: `src/data/dataset_loader.py` → `load_raw_cmapss()`

### What happens:
1. Parse whitespace-delimited text file into a pandas DataFrame with named columns
2. **Drop 6 constant sensors** (1, 5, 10, 16, 18, 19) — zero variance = no information
3. Result: DataFrame with columns `[unit_id, cycle, setting_1, setting_2, setting_3, sensor_2, sensor_3, sensor_4, sensor_6, sensor_7, sensor_8, sensor_9, sensor_11, sensor_12, sensor_13, sensor_14, sensor_15, sensor_17, sensor_20, sensor_21]`

### Before → After:
```
27 columns (raw) → 21 columns (cleaned)
```

---

## Step 3: Target Label Computation

**File**: `src/data/dataset_loader.py` → `calculate_rul()`

### What happens:
For each row in each unit:

```python
max_cycle = max(cycles for this unit)     # e.g., 192
raw_rul   = max_cycle - current_cycle     # e.g., 192 - 50 = 142
RUL       = min(raw_rul, 125)             # capped at 125 (benchmark standard)
failure_in_horizon = 1 if raw_rul <= 30 else 0  # binary failure flag
```

### Why cap at 125?
At RUL=200 vs RUL=150, the engine condition is essentially identical — both are "healthy." Capping at 125 forces the regressor to focus on the **critical end-of-life** region where predictions matter most.

### Output columns added:
- `RUL` — capped regression target
- `RUL_raw` — uncapped (for ground truth comparison)
- `failure_in_horizon` — binary classification target

---

## Step 4: Leak-Free Train/Test Split

**File**: `src/data/dataset_loader.py` → `split_by_unit()`

### What happens:
1. Get all unique `unit_id` values
2. Randomly shuffle (seed=42 for reproducibility)
3. First 20% of units → **test set** (ALL cycles from these units)
4. Remaining 80% → **training set** (ALL cycles from these units)

### Why NOT random row sampling?
If unit 1's cycle 100 goes to training and cycle 101 goes to testing, the model has already "seen" the future trajectory — **temporal data leakage**. Unit-based splitting ensures zero leakage.

```
Train: 80 units × ~200 cycles = ~16,000 rows
Test:  20 units × ~200 cycles = ~4,000 rows
```

---

## Step 5: Feature Engineering

**File**: `src/features/feature_engineering.py` → `extract_time_series_features()`

### What happens:
For each of the **16 active sensors**, within each `unit_id` group:

#### A. Rolling Statistics (96 features)
```
sensor_2_roll_mean_5   = mean of last 5 cycles of sensor_2
sensor_2_roll_mean_10  = mean of last 10 cycles
sensor_2_roll_mean_20  = mean of last 20 cycles
sensor_2_roll_std_5    = std of last 5 cycles (volatility)
sensor_2_roll_std_10   = std of last 10 cycles
sensor_2_roll_std_20   = std of last 20 cycles
... (same for all 16 sensors)
```

#### B. Baseline Deviation (16 features)
```
sensor_2_dev_baseline = sensor_2_now - sensor_2_at_cycle_1
→ Measures cumulative degradation from healthy state
```

#### C. Physics Ratios (2 features)
```
ratio_temp_press_2_7 = sensor_2 / sensor_7  (Thermal / Pressure efficiency)
ratio_temp_3_4       = sensor_3 / sensor_4  (HPC / LPT temperature ratio)
```

### Final feature vector per row:
```
16 raw + 96 rolling + 16 baseline + 2 ratios = 122 features
```

---

## Step 6: Model Training (3 Layers)

**File**: `src/train.py` → `run_training_pipeline()`

### Layer 1: Autoencoder Training
```
Input: Healthy rows only (RUL ≥ 100) → 122 features
         ↓ StandardScaler (z-score normalization)
         ↓ PyTorch Autoencoder (122 → 32 → 16 → 8 → 16 → 32 → 122)
         ↓ MSE Loss, Adam optimizer, 10 epochs
         ↓ Compute 95th percentile MSE on training data → anomaly_threshold
Output: models/autoencoder.pth (58 KB)
```

### Layer 2: XGBoost Classifier Training
```
Input: All training rows → 122 features, target: failure_in_horizon
         ↓ scale_pos_weight = neg_count / pos_count (handles imbalance)
         ↓ XGBClassifier(max_depth=5, n_estimators=50, lr=0.05)
         ↓ Evaluate: ROC-AUC, PR-AUC, F1, Precision, Recall
Output: models/xgb_classifier.json (163 KB) + _meta.npz (11 KB)
```

### Layer 3: XGBoost Regressor Training
```
Input: All training rows → 122 features, target: RUL (0-125)
         ↓ XGBRegressor(max_depth=5, n_estimators=50, lr=0.05)
         ↓ Evaluate: MAE, RMSE, R²
Output: models/xgb_regressor.json (185 KB) + _meta.npz (11 KB)
```

---

## Step 7: Real-Time Inference (Dashboard Flow)

**File**: `dashboard/app.py`

When a user selects an asset and cycle on the dashboard, this happens:

### 7.1 Data Retrieval
```python
unit_df = dataset[dataset["unit_id"] == selected_unit]  # Get all cycles for this asset
sub_raw = unit_df[unit_df["cycle"] <= selected_cycle]   # Truncate to current simulation point
```

### 7.2 Feature Engineering (on-the-fly)
```python
sub_feat = extract_time_series_features(sub_raw)  # Compute all 122 features
latest_sample = sub_feat.iloc[[-1]]               # Take only the last row (current state)
```

### 7.3 Anomaly Detection
```python
mse_score, is_anomaly = ae_model.predict(latest_sample)
# mse_score = 0.0023 (low = normal), 0.85 (high = anomaly)
# is_anomaly = True/False (based on 95th percentile threshold)
```

### 7.4 Failure Prediction
```python
fail_prob = clf_model.predict_proba(latest_sample)  # 0.73 = 73% failure risk
# Uses xgb.DMatrix + Booster.predict for safe inference
```

### 7.5 RUL Estimation
```python
pred_rul = reg_model.predict(latest_sample)  # 28.5 cycles remaining
```

### 7.6 SHAP Attribution
```python
shap_drivers = exp_model.explain_sample(latest_sample, top_k=5)
# Returns: [
#   {"feature": "sensor_3_roll_mean_20", "shap_value": +0.15, "impact": "increases_risk"},
#   {"feature": "sensor_7_dev_baseline", "shap_value": -0.08, "impact": "decreases_risk"},
#   ...
# ]
```

### 7.7 NLP Report Generation
```python
report = rep_service.generate_report(
    unit_id=1, cycle=150, failure_prob=0.73,
    predicted_rul=28.5, anomaly_score=0.85, top_drivers=shap_drivers
)
# Returns: {
#   "urgency": "HIGH — SCHEDULE MAINTENANCE THIS WEEK",
#   "maintenance_summary": "Asset #1 exhibits 73.0% probability of failure...",
#   "action_items": ["Inspect thermal insulation...", "Check compressor seals..."],
#   "root_cause_explanation": "SHAP indicates HPC Temperature driving risk..."
# }
```

### 7.8 Dashboard Rendering
All outputs rendered simultaneously:
- **4 metric cards**: failure %, RUL, anomaly score, lifecycle position
- **Telemetry chart**: 3 key sensor time-series with Plotly
- **RUL trajectory**: ground truth vs. predicted with cycle marker
- **SHAP bar chart**: top 5 feature drivers with color-coded risk
- **Maintenance report**: urgency badge + summary + action items

---

## Summary: Data Transformations at Each Stage

| Stage | Input Shape | Output Shape | Key Transformation |
|-------|------------|-------------|-------------------|
| Raw data | 100 units × 200 cycles × 27 cols | ~20,000 × 27 | Data generation |
| Cleaning | ~20,000 × 27 | ~20,000 × 21 | Drop 6 constant sensors |
| Labeling | ~20,000 × 21 | ~20,000 × 24 | Add RUL, RUL_raw, failure_in_horizon |
| Split | ~20,000 × 24 | Train: ~16K, Test: ~4K | Unit-based grouping |
| Features | ~16,000 × 24 | ~16,000 × 130 | 122 features + metadata cols |
| Autoencoder | 122 features | 1 MSE score + 1 flag | Reconstruction error |
| Classifier | 122 features | 1 probability | Failure likelihood |
| Regressor | 122 features | 1 RUL value | Remaining cycles |
| SHAP | 122 features | Top-5 drivers | Per-feature attribution |
| Report | 5 drivers + metrics | Text report | Human-readable actions |
