# Model Training Guide — Predictive Maintenance & Equipment Failure Intelligence System

## 1. How You'll Train the Model

### Step 1: Get Labeled Run-to-Failure Data
Use public run-to-failure datasets (see Section 4) that provide sensor readings for equipment tracked from healthy operation through to failure.

### Step 2: Feature Engineering
- Compute rolling window statistics (mean, std, min/max, skew) over sliding time windows per sensor.
- For vibration data: extract frequency-domain features (FFT-based bands, RMS, kurtosis).
- Label each row/window with:
  - **Time-to-failure** (for RUL regression)
  - **Failure/no-failure within a horizon** (for classification, e.g., "will it fail in the next 30 cycles?")

### Step 3: Train the Anomaly Detector (Unsupervised)
- Fit Isolation Forest / Autoencoder only on "healthy" early-life data.
- No labels required — just fit the model and validate the anomaly score/reconstruction-error threshold against known degradation periods.

### Step 4: Train the XGBoost Classifier
- **Input:** engineered features
- **Output:** failure within N cycles (binary)
- Split train/test **by unit/asset ID**, not randomly — this prevents data leakage since sequences from the same engine/asset shouldn't span both train and test sets.
- Tune hyperparameters with cross-validation (`GridSearchCV` or `Optuna`): `max_depth`, `n_estimators`, `learning_rate`, `scale_pos_weight` (to handle class imbalance, since failures are rare events).

### Step 5: Train the RUL Regressor
- Same features, but output is continuous remaining cycles/hours.
- Evaluate with RMSE and an asymmetric scoring function that penalizes *late* predictions more heavily than early ones — standard practice in predictive maintenance, since underestimating remaining life is safer than overestimating it.

### Step 6: SHAP Explainability
- No training involved — simply run `shap.TreeExplainer` on the trained XGBoost model.
- Fast: typically seconds to a few minutes, even on a full dataset.

---

## 2. How Much Time Will It Take?

XGBoost on tabular data is fast — this is one of the reasons it's the right model choice here.

| Task | Time Estimate |
|---|---|
| Feature engineering on full dataset (e.g., C-MAPSS, ~100 engines) | Minutes (Pandas) |
| Training Isolation Forest / Autoencoder | Seconds to a few minutes |
| Training XGBoost (classification + regression) | Seconds to a few minutes per model, even on CPU |
| Hyperparameter tuning (Optuna, ~50–100 trials) | 10–40 minutes |
| SHAP value computation | Seconds to a couple of minutes |
| **Total: raw data → working model** | **A few hours of total work** (most time goes into feature engineering and experimentation, not compute) |

This is nothing like training a deep learning model or an LLM — XGBoost on a dataset with a few hundred thousand rows trains in seconds to low minutes, even on a laptop CPU.

---

## 3. Is Your Laptop Enough?

**Yes, almost certainly**, for a portfolio-scale project using benchmark datasets.

- **CPU-only is fine** — XGBoost doesn't need a GPU for datasets of this size (unless you scale into millions of rows with large hyperparameter searches).
- **RAM:** 8GB is workable for standard benchmark datasets (C-MAPSS is only a few MB); 16GB is comfortable if you also run autoencoders.
- **If using an Autoencoder** (PyTorch/Keras) for anomaly detection: a GPU speeds things up but isn't required — these are small networks, and CPU training at this dataset scale still only takes minutes.

**Where a laptop *would* struggle:** if you later plug in real high-frequency industrial vibration data (sampled at kHz) across years of history and thousands of assets. That becomes a cloud/Spark-scale problem — but it's not relevant for a benchmark-dataset portfolio project.

---

## 4. Where to Get the Training Dataset

Real plant sensor data is proprietary, so use these free, realistic public datasets — the same ones used in most academic and industry predictive-maintenance demos:

1. **NASA C-MAPSS (Turbofan Engine Degradation Simulation)**
   The most widely used PdM benchmark. Multiple simulated turbofan engines run to failure with sensor readings (temperature, pressure, speed) recorded over operational cycles.
   - Best for: RUL regression, failure classification
   - Source: NASA Prognostics Data Repository / Kaggle

2. **CWRU Bearing Dataset (Case Western Reserve University)**
   Vibration data from bearings with induced faults at different severities and locations.
   - Best for: frequency-domain feature engineering, anomaly detection
   - Source: CWRU Bearing Data Center website

3. **AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository)**
   Synthetic but realistic industrial machine data (temperature, torque, rotational speed, tool wear) with failure labels.
   - Best for: straightforward classification proof-of-concept
   - Source: UCI Machine Learning Repository

4. **Microsoft Azure Predictive Maintenance Dataset (Kaggle)**
   Telemetry, error logs, and failure records for a fleet of machines.
   - Best for: fleet-wide dashboard demonstrations
   - Source: Kaggle

5. **MIMII / ToyADMOS (sound-based, optional/advanced)**
   Acoustic anomaly detection datasets, useful if extending the system to sound-based monitoring.

### Practical Recommendation
Start with **NASA C-MAPSS** for RUL + failure prediction — turbine/engine degradation is conceptually very close to compressor/turbine reliability problems in oil & gas and refining, making it the most industry-relevant starting point. Add **CWRU bearing data** to cover the vibration/frequency-feature engineering and anomaly detection part of the pipeline. Together, these two datasets cover nearly everything the system architecture requires for a working, demoable prototype.
