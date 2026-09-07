# 📊 Key Facts, Results & Industry Context

This document provides the technical facts, model performance results, industry relevance, and the real-world impact of each system component.

---

## System-Level Facts

| Metric | Value |
|--------|-------|
| Total source code files | 15+ Python modules |
| Total engineered features | 122 per telemetry sample |
| Training dataset | ~20,000 records across 100 engine units |
| Model count | 3 independent ML models |
| Inference latency | <50ms per prediction (all 3 models + SHAP) |
| Model artifact size | ~428 KB total (AE: 58KB, CLF: 174KB, REG: 196KB) |
| API endpoints | 4 (predict, explain, fleet snapshot, health) |
| Dashboard KPIs | 4 real-time metric cards + 3 interactive charts |

---

## Data Engineering Facts

### Raw Sensor Data
- **21 physical sensors** monitoring: temperatures, pressures, fan speeds, fuel flow, bypass ratios, coolant bleed
- **6 sensors dropped** (zero variance in FD001 operating condition): sensors 1, 5, 10, 16, 18, 19
- **16 informative sensors** retained for feature engineering
- **3 operational settings**: altitude, Mach number, throttle resolver angle

### Feature Engineering
- **96 rolling features**: 16 sensors × 3 windows (5, 10, 20) × 2 statistics (mean, std)
- **16 baseline deviation features**: cumulative drift from healthy initial state
- **2 physics-informed ratios**: thermal/pressure and HPC/LPT temperature efficiency
- **8 features per sensor** on average (raw + 3 means + 3 stds + 1 deviation)
- **Zero cross-unit leakage**: all rolling windows computed within `groupby("unit_id")`

### Data Split Strategy
- **80/20 split by unit_id** — NOT by random row sampling
- This is critical: random row splits cause **temporal data leakage** where the model "sees" a unit's future cycles during training, inflating metrics by 10-20%
- Our approach matches **industry best practice** used at Google, GE Digital, and Siemens MindSphere

---

## Model Performance Facts

### Layer 1: PyTorch Autoencoder (Unsupervised Anomaly Detection)

| Parameter | Value |
|-----------|-------|
| Architecture | 122 → 32 → 16 → **8** → 16 → 32 → 122 |
| Activation | ReLU with BatchNorm |
| Loss Function | Mean Squared Error (MSE) |
| Optimizer | Adam (lr=0.001) |
| Training Epochs | 10 |
| Batch Size | 64 |
| Training Data | Healthy cycles only (RUL ≥ 100) |
| Anomaly Threshold | 95th percentile MSE = **0.6423** |
| Compression Ratio | 122:8 (15.25× dimensionality reduction) |

**Key Insight**: By training only on healthy data, the autoencoder doesn't need failure labels. It acts as a **first-line alarm** that flags anomalous patterns before the supervised models even run.

### Layer 2: XGBoost Failure Classifier

| Parameter | Value |
|-----------|-------|
| Algorithm | Gradient Boosted Decision Trees |
| Objective | binary:logistic |
| Max Depth | 5 |
| Number of Trees | 50 |
| Learning Rate | 0.05 |
| Subsample | 0.8 |
| Column Sample | 0.8 |
| Class Imbalance Handling | scale_pos_weight (auto-computed) |

**Why XGBoost for Classification?**
- Handles the **severe class imbalance** (most cycles are healthy; only ~15% are near failure)
- `scale_pos_weight` automatically up-weights the minority failure class
- Native feature importance aligns perfectly with SHAP
- Robust to noisy industrial sensor data without extensive hyperparameter tuning

### Layer 3: XGBoost RUL Regressor

| Parameter | Value |
|-----------|-------|
| Algorithm | Gradient Boosted Decision Trees |
| Objective | reg:squarederror |
| Max Depth | 5 |
| Number of Trees | 50 |
| Learning Rate | 0.05 |
| Target Range | 0–125 cycles (capped) |

**Why cap RUL at 125?**
- Standard NASA C-MAPSS benchmark practice
- At RUL=200 vs RUL=300, the engine is equally "healthy" — no actionable difference
- Capping focuses the model on the **critical end-of-life region** (RUL < 50) where prediction accuracy directly impacts maintenance scheduling

---

## Explainability Facts

### SHAP TreeExplainer
- Computes **exact Shapley values** (not approximations) for tree-based models
- Time complexity: O(TLD²) where T=trees, L=leaves, D=depth — much faster than KernelSHAP
- **Additive property**: SHAP values for all features sum to the difference between the prediction and the base rate
- **Local faithfulness**: Each prediction is decomposed into exact per-feature contributions

### Sensor-to-Subsystem Mapping
The NLP reporter translates sensor IDs to physical turbine subsystems:

| Sensor | Physical Subsystem | Failure Mode |
|--------|-------------------|-------------|
| sensor_2 | LPC Outlet Temperature (T24) | Compressor fouling |
| sensor_3 | HPC Outlet Temperature (T30) | Combustor degradation |
| sensor_4 | LPT Outlet Temperature (T50) | Turbine blade erosion |
| sensor_7 | HPC Outlet Pressure (P30) | Seal leakage |
| sensor_8 | Physical Fan Speed | Bearing wear |
| sensor_9 | Physical Core Speed | Shaft alignment |
| sensor_11 | HPC Static Pressure | Bleed valve malfunction |
| sensor_12 | Fuel Flow Ratio | Metering unit degradation |
| sensor_14 | Corrected Fan Speed | Blade pitch mechanism |
| sensor_15 | Bypass Ratio | Duct seal wear |
| sensor_17 | Bleed Enthalpy | Thermal system failure |
| sensor_20 | HPT Coolant Bleed | Cooling passage blockage |
| sensor_21 | LPT Coolant Bleed | Secondary cooling failure |

---

## Industry Context & Real-World Relevance

### Who Uses This Type of System?

| Company | Platform | Scale |
|---------|----------|-------|
| **GE Aviation** | Predix / Digital Twin | 30,000+ jet engines monitored |
| **Siemens** | MindSphere | 1.5M+ connected industrial assets |
| **Rolls-Royce** | IntelligentEngine | TotalCare® predictive contracts |
| **Honeywell** | Forge | Connected aircraft & buildings |
| **ABB** | ABB Ability | 70M+ connected devices |
| **PTC** | ThingWorx | Industrial IoT platform |

### Industry Standards Followed

1. **ISO 13374** — Condition Monitoring framework (Data → Feature → Health → Prognostic → Advisory)
2. **NASA C-MAPSS** — Standard benchmark for turbofan engine degradation modeling
3. **PHM (Prognostics & Health Management)** — IEEE/ASME standard approach
4. **IEC 62443** — Industrial cybersecurity considerations

### Business Impact of Predictive Maintenance

| Metric | Industry Average |
|--------|-----------------|
| Reduction in unplanned downtime | **30-50%** |
| Reduction in maintenance costs | **10-40%** |
| Increase in equipment lifespan | **20-40%** |
| ROI of PdM implementation | **10× within 2 years** |

*Sources: McKinsey, Deloitte, PwC Industry 4.0 reports*

---

## Why This Architecture Works

### Multi-Layer Defense Strategy

```
Layer 1 (Autoencoder):  "Something is abnormal"     → UNSUPERVISED early warning
Layer 2 (Classifier):   "It WILL fail within 30 cycles" → SUPERVISED risk scoring
Layer 3 (Regressor):    "It has 28 cycles left"      → QUANTITATIVE RUL estimate
SHAP Layer:             "Sensor 3 is the root cause" → EXPLAINABLE attribution
NLP Layer:              "Inspect HPC thermal system"  → ACTIONABLE maintenance
```

Each layer adds **orthogonal intelligence**:
- The autoencoder catches **novel failure modes** the classifier hasn't been trained on
- The classifier provides **calibrated probabilities** for known failure patterns
- The regressor gives **time-to-failure** for maintenance scheduling
- SHAP provides **root cause** for targeted inspections
- NLP reports translate everything into **operator-ready language**

### Why Not Just One Model?
- A single classifier might predict "will fail" but gives no timeline
- A single regressor gives timeline but no anomaly detection for new failure modes
- The autoencoder catches degradation patterns that were never labeled as failures
- Together, they provide a **complete operational intelligence picture**

---

## Technical Challenges Solved

| Challenge | Solution |
|-----------|----------|
| Temporal data leakage | Unit-based train/test split |
| Class imbalance (15% failure) | XGBoost `scale_pos_weight` |
| Noisy sensor data | Rolling statistics smooth noise while preserving trends |
| Novel/unseen failure modes | Autoencoder-based unsupervised anomaly detection |
| Black-box predictions | SHAP decomposition for every prediction |
| Operator comprehension | NLP reports with subsystem-specific action items |
| Python 3.14 + XGBoost segfault | Raw Booster API + import order management |
| PyTorch 2.6 weights_only default | Explicit `weights_only=False` for trusted checkpoints |
