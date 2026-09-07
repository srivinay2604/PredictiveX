import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import os
import sys

# Add project root to python path for direct module imports if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# CRITICAL: Import XGBoost modules BEFORE torch-based modules to avoid segfault
# on Python 3.14 + Apple Silicon (torch + xgb.Booster.load_model conflict)
from src.models.failure_classifier import FailureClassifierXGBoost
from src.models.rul_regressor import RULRegressorXGBoost
from src.models.anomaly_detector import AnomalyDetectorAutoencoder
from src.data.dataset_loader import load_raw_cmapss, calculate_rul
from src.features.feature_engineering import extract_time_series_features, get_feature_column_names
from src.models.explainer import ModelExplainer
from src.llm.reporter import MaintenanceReportGenerator

st.set_page_config(
    page_title="Asset Health & Predictive Maintenance Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Glassmorphism & High-End Industrial Styling
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    .metric-card {
        background: rgba(30, 34, 45, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 22px 18px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        min-height: 160px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }
    .metric-label {
        color: #9CA3AF;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin: 0;
    }
    .metric-value {
        color: #00E5FF;
        font-size: 28px;
        font-weight: 700;
        font-family: 'Courier New', monospace;
        margin: 4px 0;
    }
    .metric-sub {
        color: #6B7280;
        font-size: 12px;
        margin: 0;
    }
    .status-badge-critical {
        background-color: #FF2B2B;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
    }
    .status-badge-high {
        background-color: #FF8800;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
    }
    .status-badge-medium {
        background-color: #FFBB00;
        color: black;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
    }
    .status-badge-low {
        background-color: #00C851;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
    }
    .action-item {
        background: #1A2234;
        border-left: 4px solid #00E5FF;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 4px;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_all_pipeline_models():
    if not os.path.exists("models/autoencoder.pth") or not os.path.exists("models/xgb_classifier.json"):
        st.error("Model artifacts missing! Running training pipeline first...")
        from src.train import run_training_pipeline
        run_training_pipeline()
    
    # CRITICAL: Load XGBoost models BEFORE PyTorch autoencoder to avoid segfault
    clf = FailureClassifierXGBoost().load("models/xgb_classifier.json")
    reg = RULRegressorXGBoost().load("models/xgb_regressor.json")
    ae = AnomalyDetectorAutoencoder().load("models/autoencoder.pth")
    exp = ModelExplainer(clf)
    rep = MaintenanceReportGenerator()
    return ae, clf, reg, exp, rep

@st.cache_data
def load_cached_dataset():
    from src.data.download_data import download_cmapss_data
    path = download_cmapss_data()
    raw = load_raw_cmapss(path)
    labeled = calculate_rul(raw)
    return labeled

# Application Header
st.title("⚡ Predictive Maintenance & Equipment Failure Intelligence System")
st.markdown("**Real-Time IIoT Telemetry Analytics, Anomaly Detection, XGBoost Failure Prediction & SHAP Root-Cause Diagnostics**")

try:
    ae_model, clf_model, reg_model, exp_model, rep_service = load_all_pipeline_models()
    dataset = load_cached_dataset()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# Sidebar Controls
st.sidebar.header("🎛️ Asset Simulation Controls")

units = sorted(dataset["unit_id"].unique())
selected_unit = st.sidebar.selectbox("Select Equipment Asset (Unit ID)", units, index=0)

unit_df = dataset[dataset["unit_id"] == selected_unit].sort_values(by="cycle").reset_index(drop=True)
max_cycles = len(unit_df)

selected_cycle = st.sidebar.slider(
    "Simulate Telemetry Cycle",
    min_value=1,
    max_value=max_cycles,
    value=min(120, max_cycles),
    step=1,
    help="Move slider forward to simulate continuous operational degradation over time."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Architecture & Status")
st.sidebar.success("✅ PyTorch Autoencoder: Active")
st.sidebar.success("✅ XGBoost Classifier: Active")
st.sidebar.success("✅ XGBoost RUL Regressor: Active")
st.sidebar.success("✅ SHAP TreeExplainer: Active")

# Filter telemetry history up to selected_cycle
sub_raw = unit_df[unit_df["cycle"] <= selected_cycle].copy()
sub_feat = extract_time_series_features(sub_raw)
latest_sample = sub_feat.iloc[[-1]].copy()

# Run Predictions
mse_score, is_anom_flag = ae_model.predict(latest_sample)
anomaly_mse = float(mse_score[0])
is_anomaly = bool(is_anom_flag[0])

fail_prob = float(clf_model.predict_proba(latest_sample)[0])
pred_rul = float(reg_model.predict(latest_sample)[0])
actual_rul = float(latest_sample["RUL_raw"].values[0]) if "RUL_raw" in latest_sample.columns else max_cycles - selected_cycle

shap_drivers = exp_model.explain_sample(latest_sample, top_k=5)

# Metrics Bar Layout
col1, col2, col3, col4 = st.columns(4)

# Determine badge for failure risk
if fail_prob >= 0.80:
    risk_badge = '<span class="status-badge-critical">CRITICAL RISK</span>'
elif fail_prob >= 0.50:
    risk_badge = '<span class="status-badge-high">HIGH RISK</span>'
elif fail_prob >= 0.25:
    risk_badge = '<span class="status-badge-medium">MEDIUM RISK</span>'
else:
    risk_badge = '<span class="status-badge-low">HEALTHY</span>'

# Determine badge for anomaly
if is_anomaly:
    anomaly_badge = '<span class="status-badge-critical">ANOMALY DETECTED</span>'
else:
    anomaly_badge = '<span class="status-badge-low">NORMAL STATE</span>'

with col1:
    st.markdown(f'''
    <div class="metric-card">
        <p class="metric-label">FAILURE RISK PROBABILITY</p>
        <p class="metric-value">{fail_prob:.1%}</p>
        {risk_badge}
    </div>
    ''', unsafe_allow_html=True)

with col2:
    st.markdown(f'''
    <div class="metric-card">
        <p class="metric-label">PREDICTED RUL (REMAINING CYCLES)</p>
        <p class="metric-value">{pred_rul:.0f} Cycles</p>
        <p class="metric-sub">Actual Remaining: {actual_rul:.0f} Cycles</p>
    </div>
    ''', unsafe_allow_html=True)

with col3:
    st.markdown(f'''
    <div class="metric-card">
        <p class="metric-label">AUTOENCODER ANOMALY SCORE</p>
        <p class="metric-value">{anomaly_mse:.4f}</p>
        {anomaly_badge}
    </div>
    ''', unsafe_allow_html=True)

with col4:
    st.markdown(f'''
    <div class="metric-card">
        <p class="metric-label">CURRENT ASSET LIFECYCLE</p>
        <p class="metric-value">{selected_cycle} / {max_cycles}</p>
        <p class="metric-sub">Asset #{selected_unit} Operational Step</p>
    </div>
    ''', unsafe_allow_html=True)


st.markdown("---")

# ==============================================================================
# SECTION 1: 📈 MULTI-SENSOR TELEMETRY & DEGRADATION TREND (FULL WIDTH)
# ==============================================================================
st.subheader("📈 Multi-Sensor Operational Telemetry & Degradation Trend")

sensor_name_map = {
    "sensor_2": "LPC Total Temp (T24)",
    "sensor_3": "HPC Total Temp (T30)",
    "sensor_4": "LPT Total Temp (T50)",
    "sensor_7": "HPC Pressure (P30)",
    "sensor_11": "HPC Speed (Nhc)",
    "sensor_12": "Physical Fan Speed (Nf)",
    "sensor_15": "Bypass Ratio (BPR)",
    "sensor_17": "High-Pressure Turbine Cool Bleed",
    "sensor_20": "HPT Bleed Pressure",
    "sensor_21": "LPT Bleed Pressure"
}

ctrl_col1, ctrl_col2 = st.columns([3, 1])

with ctrl_col1:
    selected_sensors = st.multiselect(
        "Select Telemetry Sensors to Plot:",
        options=list(sensor_name_map.keys()),
        default=["sensor_2", "sensor_3", "sensor_4", "sensor_11", "sensor_12"],
        format_func=lambda x: f"{x} - {sensor_name_map[x]}"
    )

with ctrl_col2:
    scale_mode = st.radio(
        "Display Mode:",
        options=["Raw Values", "Normalized (Z-Score)"],
        index=0
    )

if not selected_sensors:
    selected_sensors = ["sensor_2", "sensor_3", "sensor_4"]

fig_sensors = go.Figure()
palette = ["#00E5FF", "#FF007F", "#FFD700", "#00C851", "#FF8800", "#AA66CC", "#FF4444", "#33B5E5"]

for idx, s_col in enumerate(selected_sensors):
    s_name = sensor_name_map.get(s_col, s_col)
    y_vals = sub_raw[s_col]
    if scale_mode == "Normalized (Z-Score)":
        std_val = sub_raw[s_col].std()
        if std_val > 0:
            y_vals = (sub_raw[s_col] - sub_raw[s_col].mean()) / std_val
        else:
            y_vals = sub_raw[s_col] - sub_raw[s_col].mean()

    fig_sensors.add_trace(go.Scatter(
        x=sub_raw["cycle"],
        y=y_vals,
        name=s_name,
        mode="lines",
        line=dict(color=palette[idx % len(palette)], width=2.5),
        hovertemplate=f"Cycle: %{{x}}<br>{s_name}: %{{y:.2f}}<extra></extra>"
    ))

fig_sensors.add_vline(x=selected_cycle, line_width=2, line_dash="dash", line_color="#FF2B2B", annotation_text=f"Current Cycle ({selected_cycle})", annotation_position="top left")

fig_sensors.update_layout(
    template="plotly_dark",
    height=420,
    margin=dict(l=40, r=40, t=30, b=40),
    xaxis=dict(title="Operational Cycle (Time)", showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
    yaxis=dict(title="Sensor Telemetry Amplitude" if scale_mode == "Raw Values" else "Normalized Z-Score", showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
)
st.plotly_chart(fig_sensors, use_container_width=True)

with st.expander("📊 View Raw Telemetry Data Table & Download"):
    st.dataframe(sub_raw[["unit_id", "cycle"] + selected_sensors], use_container_width=True)
    csv_data = sub_raw.to_csv(index=False)
    st.download_button(
        label="📥 Download Telemetry CSV",
        data=csv_data,
        file_name=f"unit_{selected_unit}_cycle_{selected_cycle}_telemetry.csv",
        mime="text/csv"
    )

st.markdown("---")

# ==============================================================================
# SECTION 2: 📉 REMAINING USEFUL LIFE (RUL) DECAY TRAJECTORY (FULL WIDTH)
# ==============================================================================
st.subheader("📉 Remaining Useful Life (RUL) Decay Trajectory & Failure Forecast")

full_feat = extract_time_series_features(unit_df)
full_ruls = reg_model.predict(full_feat)

fig_rul = go.Figure()

# Ground Truth RUL
fig_rul.add_trace(go.Scatter(
    x=unit_df["cycle"],
    y=unit_df["RUL_raw"],
    name="Actual Ground Truth RUL",
    line=dict(color="rgba(255, 255, 255, 0.4)", width=2, dash="dash"),
    hovertemplate="Cycle: %{x}<br>Ground Truth RUL: %{y:.0f} cycles<extra></extra>"
))

# Predicted RUL
fig_rul.add_trace(go.Scatter(
    x=unit_df["cycle"],
    y=full_ruls,
    name="XGBoost Predicted RUL",
    line=dict(color="#00C851", width=3.5),
    hovertemplate="Cycle: %{x}<br>Predicted RUL: %{y:.1f} cycles<extra></extra>"
))

# Critical Threshold Line at RUL=30 cycles
fig_rul.add_hline(y=30, line_width=2, line_dash="dot", line_color="#FF8800", annotation_text="Maintenance Warning Threshold (30 Cycles)", annotation_position="bottom right")

# Vertical marker for selected cycle
fig_rul.add_vline(x=selected_cycle, line_width=2, line_dash="solid", line_color="#00E5FF", annotation_text=f"Selected Cycle {selected_cycle}", annotation_position="top right")

fig_rul.update_layout(
    template="plotly_dark",
    height=400,
    margin=dict(l=40, r=40, t=30, b=40),
    xaxis=dict(title="Operational Cycle", showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
    yaxis=dict(title="Remaining Useful Life (Cycles)", showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
)
st.plotly_chart(fig_rul, use_container_width=True)

st.markdown("---")

# ==============================================================================
# SECTION 3: 🔍 SHAP ROOT-CAUSE FEATURE ATTRIBUTION (FULL WIDTH)
# ==============================================================================
st.subheader("🔍 SHAP Root-Cause Feature Attribution & Diagnostics")

if shap_drivers:
    driver_df = pd.DataFrame(shap_drivers)
    # Sort for best display in horizontal bar chart
    driver_df = driver_df.sort_values(by="shap_value", ascending=True)

    fig_shap = px.bar(
        driver_df,
        x="shap_value",
        y="feature",
        orientation="h",
        color="shap_value",
        color_continuous_scale=["#00C851", "#FF8800", "#FF2B2B"],
        title="Physical Sensor Drivers Increasing Failure Risk (SHAP Impact)"
    )
    fig_shap.update_layout(
        template="plotly_dark",
        height=380,
        margin=dict(l=150, r=40, t=40, b=40),
        xaxis_title="SHAP Value (Contribution to Failure Risk)",
        yaxis_title="Sensor Feature",
        coloraxis_showscale=False
    )
    fig_shap.update_yaxes(tickfont=dict(size=13))
    st.plotly_chart(fig_shap, use_container_width=True)

    # Interactive Feature Inspector
    with st.expander("🔎 Interactive Feature Inspector & Statistical Breakdown"):
        inspect_feat = st.selectbox("Select Feature to Inspect:", driver_df["feature"].tolist())
        feat_info = driver_df[driver_df["feature"] == inspect_feat].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Feature Name", str(feat_info["feature"]))
        val = feat_info["value"] if "value" in feat_info else feat_info.get("feature_value", 0.0)
        c2.metric("Current Value", f"{val:.4f}")
        c3.metric("SHAP Contribution", f"{feat_info['shap_value']:+.4f}")

st.markdown("---")

# ==============================================================================
# SECTION 4: 🤖 NATURAL LANGUAGE MAINTENANCE REPORT (FULL WIDTH)
# ==============================================================================
st.subheader("🤖 Natural Language Maintenance Report & Executive Action Plan")

report_dict = rep_service.generate_report(
    unit_id=selected_unit,
    cycle=selected_cycle,
    failure_prob=fail_prob,
    predicted_rul=pred_rul,
    anomaly_score=anomaly_mse,
    top_drivers=shap_drivers
)

# Header Alert Card
urgency = report_dict['urgency']
if "CRITICAL" in urgency:
    st.error(f"🚨 **Urgency Level:** {urgency}")
elif "HIGH" in urgency:
    st.warning(f"⚠️ **Urgency Level:** {urgency}")
elif "MEDIUM" in urgency:
    st.warning(f"🔔 **Urgency Level:** {urgency}")
else:
    st.success(f"✅ **Urgency Level:** {urgency}")

st.markdown(f"### 📋 Executive Summary\n{report_dict['maintenance_summary']}")

st.markdown("### 🔧 Recommended Interactive Maintenance Action Items")
st.caption("Check off items as field technicians complete maintenance work orders:")

for idx, action in enumerate(report_dict["action_items"]):
    st.checkbox(f"Work Order #{idx+1}: {action}", key=f"wo_{selected_unit}_{selected_cycle}_{idx}")

with st.expander("📖 Detailed Root Cause & System Breakdown"):
    st.markdown(report_dict["root_cause_explanation"])

# Downloadable Maintenance Report
import json

def json_default(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

json_report = json.dumps(report_dict, indent=2, default=json_default)
st.download_button(
    label="📄 Download Official Executive Maintenance Report (JSON)",
    data=json_report,
    file_name=f"maintenance_report_unit_{selected_unit}_cycle_{selected_cycle}.json",
    mime="application/json"
)

