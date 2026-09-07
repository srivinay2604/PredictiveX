# 🔄 Phase-Wise Implementation Guide

This document explains the **5 phases** of the project in detail — what happens in each phase, what the steps are, and how they connect to form the complete system.

---

## Phase 1: Data Ingestion & Preprocessing

### Goal
Acquire industrial sensor data, clean it, compute target labels, and partition it for leak-free model training.

### Steps

#### Step 1.1: Data Source — NASA C-MAPSS Benchmark
- **Source File**: `src/data/download_data.py`
- The system uses the **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** turbofan engine degradation dataset — the gold standard benchmark in predictive maintenance research.
- Contains **21 sensor channels** per engine unit, recorded every operational cycle until failure.
- If the public dataset is unavailable, a **synthetic data generator** creates physics-realistic run-to-failure profiles with:
  - Exponential degradation curves (quadratic wear function)
  - Gaussian sensor noise at realistic amplitudes
  - 100 engine units × ~200 cycles each = ~20,000 telemetry records

#### Step 1.2: Data Loading & Static Sensor Removal
- **Source File**: `src/data/dataset_loader.py` → `load_raw_cmapss()`
- Loads the whitespace-delimited sensor file into a structured pandas DataFrame.
- **Drops 6 constant sensors** (sensors 1, 5, 10, 16, 18, 19) that have zero variance and carry no degradation signal — standard C-MAPSS preprocessing.
- Retains **16 active sensors** with operational settings.

#### Step 1.3: Target Label Engineering
- **Source File**: `src/data/dataset_loader.py` → `calculate_rul()`
- **RUL (Remaining Useful Life)**: Computed as `max_cycle - current_cycle` per unit, capped at 125 cycles (standard benchmark).
- **failure_in_horizon**: Binary flag = 1 if RUL ≤ 30 cycles (failure imminent within 30 cycles).
- This gives us **two supervised learning targets**: a continuous regression target (RUL) and a binary classification target (failure_in_horizon).

#### Step 1.4: Unit-Based Train/Test Split
- **Source File**: `src/data/dataset_loader.py` → `split_by_unit()`
- **Critical design decision**: Split by entire `unit_id` groups (80% train / 20% test), NOT by random row sampling.
- This prevents **temporal data leakage** — no test unit's future data appears in training, which would inflate performance metrics artificially.
- Result: ~80 training units, ~20 test units.

### Key Facts
- 21 raw sensors → 16 active sensors after constant removal
- ~20,000 telemetry records across 100 engine units
- 2 target variables: RUL (continuous) + failure_in_horizon (binary)
- Split ratio: 80/20 by unit_id group

---

## Phase 2: Signal Processing & Feature Engineering

### Goal
Transform raw sensor readings into ML-ready features that capture degradation patterns, temporal trends, and physics-informed relationships.

### Steps

#### Step 2.1: Rolling Window Statistics
- **Source File**: `src/features/feature_engineering.py` → `extract_time_series_features()`
- For each of the 16 active sensors, compute:
  - **Rolling Mean** over windows of 5, 10, 20 cycles → smooths noise, reveals trend
  - **Rolling Std** over windows of 5, 10, 20 cycles → captures increasing vibration/instability
- All rolling operations are **partitioned by unit_id** using `groupby().transform()` to prevent cross-unit contamination.
- Produces: 16 sensors × 3 windows × 2 stats = **96 rolling features**

#### Step 2.2: Baseline Deviation Features
- For each sensor, compute the **deviation from cycle 1** (healthy baseline).
- Formula: `sensor_current - sensor_cycle_1` per unit
- This directly measures **cumulative degradation** from the initial healthy state.
- Produces: 16 × 1 = **16 baseline deviation features**

#### Step 2.3: Physics-Informed Interaction Ratios
- **Thermal-to-Pressure Ratio**: `sensor_2 (LPC Temp) / sensor_7 (HPC Pressure)`
  - Captures thermodynamic efficiency degradation
- **HPC/LPT Temperature Ratio**: `sensor_3 (HPC Temp) / sensor_4 (LPT Temp)`
  - Captures compressor-turbine energy transfer efficiency
- These are **domain-engineered features** based on gas turbine physics.
- Produces: **2 interaction ratio features**

### Feature Summary

| Feature Category | Count | Purpose |
|-----------------|-------|---------|
| Raw sensor readings | 16 | Base signal |
| Rolling means (5, 10, 20) | 48 | Trend smoothing |
| Rolling stds (5, 10, 20) | 48 | Instability detection |
| Baseline deviations | 16 | Cumulative drift |
| Physics ratios | 2 | Domain knowledge |
| **Total** | **122** | **ML input features** |

---

## Phase 3: Multi-Layer ML Model Training

### Goal
Train three independent ML models that provide orthogonal failure intelligence — anomaly detection, failure classification, and RUL regression.

### Steps

#### Step 3.1: Layer 1 — PyTorch Autoencoder (Unsupervised Anomaly Detection)
- **Source File**: `src/models/anomaly_detector.py`
- **Training Data**: Only "healthy" cycles where RUL ≥ 100 (top 80% of life)
- **Architecture**:
  - Encoder: `122 → 32 → 16 → 8` (with BatchNorm + ReLU)
  - Decoder: `8 → 16 → 32 → 122` (symmetric)
  - Latent dimension: 8 (compressed representation)
- **Training**: Adam optimizer, MSE loss, 10 epochs, batch size 64
- **Anomaly Threshold**: 95th percentile of reconstruction MSE on healthy training data
- **Inference**: If reconstruction MSE > threshold → **anomaly flag raised**
- **Key Insight**: The autoencoder learns what "healthy" looks like. When degraded data arrives, it cannot reconstruct it well → high MSE → anomaly.

#### Step 3.2: Layer 2 — XGBoost Classifier (Supervised Failure Prediction)
- **Source File**: `src/models/failure_classifier.py`
- **Target**: `failure_in_horizon` (binary: will fail within 30 cycles?)
- **Handles class imbalance**: `scale_pos_weight = negative_count / positive_count`
- **Hyperparameters**: max_depth=5, n_estimators=50, learning_rate=0.05, subsample=0.8
- **Metrics**: ROC-AUC, PR-AUC, F1, Precision, Recall
- **Output**: Probability [0.0 → 1.0] of imminent failure

#### Step 3.3: Layer 3 — XGBoost Regressor (RUL Estimation)
- **Source File**: `src/models/rul_regressor.py`
- **Target**: `RUL` (continuous: how many cycles until failure?)
- **Hyperparameters**: max_depth=5, n_estimators=50, learning_rate=0.05
- **Metrics**: MAE (Mean Absolute Error), RMSE, R²
- **Output**: Predicted remaining useful life in cycles (0–125)

#### Step 3.4: SHAP Explainability
- **Source File**: `src/models/explainer.py`
- Uses **SHAP TreeExplainer** on the XGBoost classifier
- For every prediction, computes the **exact contribution of each feature** to the failure risk score
- Returns top-K drivers with feature name, SHAP value, and impact direction
- **Key Insight**: Operators don't just want "this will fail" — they need to know **which sensor is driving the risk** so they can target the right subsystem for inspection.

### Training Pipeline
- **Source File**: `src/train.py` → `run_training_pipeline()`
- Orchestrates the entire flow: data → features → train all 3 models → save artifacts
- Configuration-driven via `config/config.yaml`
- Produces model artifacts in `models/` directory

---

## Phase 4: Production Serving (API + Dashboard)

### Goal
Make the trained models accessible for real-time inference via a REST API and an interactive visual dashboard.

### Steps

#### Step 4.1: FastAPI REST API
- **Source File**: `src/api/main.py`
- **Endpoints**:
  - `POST /predict/telemetry` — Send sensor history, get anomaly score + failure probability + RUL + SHAP drivers
  - `POST /explain/recommendation` — Get natural language maintenance report
  - `GET /fleet/snapshot` — Get fleet-wide equipment status summary
  - `GET /health` — Service health check
- Models are loaded into memory at startup for sub-millisecond inference
- CORS enabled for cross-origin dashboard/frontend access

#### Step 4.2: Streamlit Operations Dashboard
- **Source File**: `dashboard/app.py`
- **Features**:
  - 4 KPI metric cards: Failure Risk, Predicted RUL, Anomaly Score, Asset Lifecycle
  - Multi-sensor telemetry time-series chart (Plotly)
  - RUL decay trajectory: ground truth vs. XGBoost prediction
  - SHAP feature attribution bar chart
  - Natural language maintenance report with action items
  - Sidebar: Asset selector + Cycle simulation slider
- **Glassmorphism UI**: Dark theme, custom CSS, responsive layout
- Real-time updates when user adjusts the simulation slider

#### Step 4.3: NLP Maintenance Report Generator
- **Source File**: `src/llm/reporter.py`
- Translates numeric ML outputs into **structured engineering maintenance notes**
- Maps sensor IDs to human-readable turbine subsystem names (LPC Temp, HPC Pressure, etc.)
- Generates:
  - Urgency classification (CRITICAL / HIGH / MEDIUM / LOW)
  - Maintenance summary with risk percentages
  - Physics-specific action items (inspect thermal liners, check compressor seals, etc.)
  - Root cause explanation from SHAP decomposition

---

## Phase 5: Testing, Validation & Deployment Readiness

### Goal
Ensure model correctness, prevent regressions, and validate the full pipeline end-to-end.

### Steps

#### Step 5.1: Unit Testing
- **Directory**: `tests/`
- Tests cover:
  - Data loading and preprocessing correctness
  - Feature engineering output shape and column names
  - Model save/load roundtrip verification
  - Prediction output format and value ranges
  - API endpoint response schemas

#### Step 5.2: Model Validation
- Train/test split prevents data leakage
- Classification metrics: ROC-AUC, PR-AUC, F1
- Regression metrics: MAE, RMSE, R²
- Anomaly detection: test set anomaly ratio validation

#### Step 5.3: End-to-End Pipeline Verification
- Run `src/train.py` → verify all 3 model artifacts saved
- Launch dashboard → verify all charts render with live predictions
- Test API endpoints → verify JSON response schema compliance

#### Step 5.4: Production Hardening
- Centralized YAML configuration (no hardcoded hyperparameters)
- Graceful error handling in API and dashboard
- Model artifact versioning via filesystem
- Streamlit cache management (`@st.cache_resource`, `@st.cache_data`)
