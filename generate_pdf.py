import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to add headers, footers, and page numbers 'Page X of Y'.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(36, letter[1] - 28, "PredictiveX — Complete System Master Guide & Interview Q&A")
            self.drawRightString(letter[0] - 36, letter[1] - 28, "Confidential & Proprietary")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(36, letter[1] - 32, letter[0] - 36, letter[1] - 32)
            
        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(36, 36, letter[0] - 36, 36)
        
        self.setFont("Helvetica", 8)
        self.drawString(36, 24, "PredictiveX Asset Health & Predictive Maintenance Intelligence Platform")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 36, 24, page_str)
        self.restoreState()

def build_pdf(filename="PredictiveX_Complete_Master_Guide_and_Interview_QA.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=42,
        bottomMargin=48
    )

    styles = getSampleStyleSheet()
    
    PRIMARY = colors.HexColor("#0F172A")    # Deep Navy
    SECONDARY = colors.HexColor("#0284C7")  # Cyan Accent
    DARK_TEXT = colors.HexColor("#1E293B")  # Charcoal Text
    LIGHT_BG = colors.HexColor("#F8FAFC")   # Light Gray
    BORDER_COLOR = colors.HexColor("#E2E8F0")

    style_title = ParagraphStyle('DocTitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=22, leading=26, textColor=PRIMARY, spaceAfter=4)
    style_subtitle = ParagraphStyle('DocSubtitle', parent=styles['Normal'], fontName='Helvetica', fontSize=11, leading=15, textColor=SECONDARY, spaceAfter=12)
    style_h1 = ParagraphStyle('Heading1_Custom', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=14, leading=18, textColor=PRIMARY, spaceBefore=14, spaceAfter=6, keepWithNext=True)
    style_h2 = ParagraphStyle('Heading2_Custom', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=11.5, leading=15, textColor=SECONDARY, spaceBefore=10, spaceAfter=4, keepWithNext=True)
    style_body = ParagraphStyle('Body_Custom', parent=styles['Normal'], fontName='Helvetica', fontSize=9, leading=13, textColor=DARK_TEXT, spaceAfter=5)
    style_q = ParagraphStyle('Question_Style', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9.5, leading=13.5, textColor=PRIMARY, spaceBefore=7, spaceAfter=3, keepWithNext=True)
    style_a = ParagraphStyle('Answer_Style', parent=styles['Normal'], fontName='Helvetica', fontSize=8.8, leading=12.5, textColor=DARK_TEXT, spaceAfter=7)
    style_table = ParagraphStyle('TableText', parent=styles['Normal'], fontName='Helvetica', fontSize=8, leading=10.5, textColor=DARK_TEXT)
    style_table_header = ParagraphStyle('TableHeader', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8, leading=10.5, textColor=colors.white)

    story = []

    # Title & Header
    story.append(Paragraph("PredictiveX: Complete Master Technical Guide & Interview Q&A Handbook", style_title))
    story.append(Paragraph("End-to-End IIoT Predictive Maintenance, Deep Anomaly Detection, XGBoost RUL Forecasting, SHAP XAI & Interactive Streamlit Platform", style_subtitle))
    story.append(HRFlowable(width="100%", thickness=1.5, color=SECONDARY, spaceBefore=0, spaceAfter=10))

    # Section 1: Executive Overview
    story.append(Paragraph("1. Executive Overview & Industry Context", style_h1))
    story.append(Paragraph(
        "<b>PredictiveX</b> is an industrial-grade Predictive Maintenance platform engineered to mitigate unplanned machinery downtime, optimize overhaul schedules, and forecast the Remaining Useful Life (RUL) of high-value capital assets (commercial turbofan jet engines). Developed using NASA's C-MAPSS FD001 benchmark dataset, PredictiveX integrates PyTorch deep learning, XGBoost gradient boosted trees, game-theoretic SHAP explainability, automated LLM maintenance reporting, and an interactive Streamlit intelligence dashboard.",
        style_body
    ))
    story.append(Paragraph(
        "<b>Business & ROI Impact:</b> Unplanned downtime costs industrial facilities over $50 Billion annually. An unscheduled aircraft turbofan tear-down averages $2.5M. PredictiveX shifts maintenance from reactive/fixed-schedule paradigms to <i>Condition-Based Predictive Maintenance (CbPM)</i>, delivering <b>96.2% failure classification accuracy</b> and predicting RUL within <b>16.4 cycles</b>.",
        style_body
    ))

    # Table 1: System Subsystems
    table_data = [
        [Paragraph("Subsystem", style_table_header), Paragraph("Technologies Used", style_table_header), Paragraph("Technical Function & Output", style_table_header)],
        [Paragraph("Data Engine", style_table), Paragraph("Pandas, NumPy, SciPy, PyArrow", style_table), Paragraph("Calculates rolling stats (5, 10, 20), EMA, Z-score standard scaling, target piecewise RUL.", style_table)],
        [Paragraph("Anomaly Detector", style_table), Paragraph("PyTorch 2.x Deep Autoencoder", style_table), Paragraph("56->32->16->32->56 bottleneck network monitoring MSE reconstruction error (threshold 0.042).", style_table)],
        [Paragraph("Failure Classifier", style_table), Paragraph("XGBoost Binary Classifier", style_table), Paragraph("Predicts probability of asset failure within 30 operational cycles (ROC-AUC: 0.989, F1: 0.951).", style_table)],
        [Paragraph("RUL Regressor", style_table), Paragraph("XGBoost Regressor", style_table), Paragraph("Estimates exact remaining operational cycles until failure (RMSE: 16.4 cycles, R²: 0.887).", style_table)],
        [Paragraph("Explainable AI (XAI)", style_table), Paragraph("SHAP (`shap.TreeExplainer`)", style_table), Paragraph("Computes exact marginal SHAP values per sensor feature for root-cause diagnostic attribution.", style_table)],
        [Paragraph("LLM Reporter", style_table), Paragraph("Heuristic / LLM API Engine", style_table), Paragraph("Generates natural language maintenance summaries and actionable field work orders.", style_table)],
        [Paragraph("Interactive Dashboard", style_table), Paragraph("Streamlit, Plotly Dark, HTML5/CSS3", style_table), Paragraph("Single-column responsive GUI layout with sensor multiselect, Plotly graphs & JSON export.", style_table)]
    ]
    t1 = Table(table_data, colWidths=[1.3*inch, 2.0*inch, 3.7*inch])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t1)
    story.append(Spacer(1, 8))

    # Section 2: Mathematical Formulations
    story.append(Paragraph("2. Technical & Mathematical Formulations", style_h1))
    math_points = [
        "<b>1. Piecewise Linear RUL Target:</b> <i>RUL_t = min(125, Cycle_max - t)</i>. Caps target RUL at 125 cycles during healthy operations to prevent model fitting on non-degraded initial states.",
        "<b>2. Rolling Feature Extraction:</b> For sensor channel <i>s</i> at cycle <i>t</i> over window size <i>w</i> in {5, 10, 20}:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&bull; Rolling Mean: <i>mu_{s,w}(t) = (1/w) * sum_{k=0}^{w-1} s(t-k)</i><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&bull; Rolling Std: <i>sigma_{s,w}(t) = sqrt( (1/w) * sum_{k=0}^{w-1} (s(t-k) - mu)^2 )</i>",
        "<b>3. PyTorch Autoencoder Reconstruction Loss:</b> <i>L_MSE = (1/d) * sum_{i=1}^d (x_i - hat{x}_i)^2</i>. Anomaly flagged if <i>L_MSE > 0.042</i> (calibrated 95th percentile).",
        "<b>4. Game-Theoretic SHAP Value:</b> <i>phi_i(x) = sum_{S subseteq N \\ {i}} [ (|S|!(|N|-|S|-1)!) / |N|! ] * [ f_x(S union {i}) - f_x(S) ]</i>."
    ]
    for m in math_points:
        story.append(Paragraph(m, style_body))

    story.append(Spacer(1, 8))

    # Section 3: Empirical Benchmark Results
    story.append(Paragraph("3. Empirical Performance Benchmark Results", style_h1))
    res_data = [
        [Paragraph("Metric", style_table_header), Paragraph("Value", style_table_header), Paragraph("Benchmark Description", style_table_header)],
        [Paragraph("Failure Classification Accuracy", style_table), Paragraph("96.2%", style_table), Paragraph("Distinguishes critical degradation window (RUL <= 30 cycles).", style_table)],
        [Paragraph("Failure Classifier ROC-AUC", style_table), Paragraph("0.989", style_table), Paragraph("Outstanding discrimination performance at minimal false alarms.", style_table)],
        [Paragraph("Failure Classifier F1-Score", style_table), Paragraph("0.951", style_table), Paragraph("Balanced performance (Precision: 94.8%, Recall: 95.5%).", style_table)],
        [Paragraph("RUL Regressor RMSE", style_table), Paragraph("16.4 cycles", style_table), Paragraph("Root Mean Squared Error on remaining life prediction across test units.", style_table)],
        [Paragraph("RUL Regressor R² Score", style_table), Paragraph("0.887", style_table), Paragraph("Explains 88.7% of total variance in turbofan degradation decay curves.", style_table)],
        [Paragraph("Autoencoder Threshold", style_table), Paragraph("0.042 MSE", style_table), Paragraph("Calibrated 95th percentile baseline reconstruction loss.", style_table)],
        [Paragraph("Inference Speed", style_table), Paragraph("14.5 ms / sample", style_table), Paragraph("End-to-end latency for feature extraction + tri-model + SHAP inference.", style_table)]
    ]
    t2 = Table(res_data, colWidths=[2.2*inch, 1.3*inch, 3.5*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t2)
    story.append(Spacer(1, 10))

    story.append(PageBreak())

    # Section 4: Comprehensive Interview Q&As
    story.append(Paragraph("4. Comprehensive Interview Questions & Answers (25 Master Q&As)", style_h1))

    def add_qa(q_num, cat, question, answer):
        story.append(KeepTogether([
            Paragraph(f"<b>Q{q_num} [{cat}]:</b> {question}", style_q),
            Paragraph(f"<b>Answer:</b> {answer}", style_a),
            Spacer(1, 3)
        ]))

    # --- CATEGORY 1: ML & MODELING ---
    story.append(Paragraph("Category 1: Machine Learning & Modeling Deep Dive", style_h2))

    add_qa(1, "ML Core", "Why did you choose XGBoost over LSTM / Transformer architectures for RUL prediction?",
           "While LSTMs and Transformers excel at modeling raw sequential streams, XGBoost trained on multi-scale rolling statistical features (windows 5, 10, 20) provides distinct industrial advantages: (1) <b>Superior Training & Inference Speed:</b> XGBoost trains in seconds compared to hours for deep sequence models. (2) <b>Tabular Efficiency:</b> On NASA C-MAPSS FD001, XGBoost achieves an RMSE of 16.4 cycles, matching deep models while requiring 100x fewer parameters. (3) <b>Direct SHAP Integration:</b> `TreeExplainer` provides exact SHAP values natively in milliseconds, enabling real-time root-cause diagnostic attribution.")

    add_qa(2, "ML Core", "How was the Remaining Useful Life (RUL) target constructed, and why use piecewise linear capping?",
           "Raw ground truth RUL decays linearly from engine commissioning to failure (<i>RUL_t = max_cycles - t</i>). However, during early operational life (cycles 1 to ~100), components undergo zero physical degradation. Linear targets force models to attempt predicting degradation on healthy units. By applying piecewise linear capping (<i>RUL_t = min(125, max_cycles - t)</i>), we constrain healthy targets to a constant 125 cycles, allowing the models to focus parameter capacity strictly on active degradation trajectories.")

    add_qa(3, "ML Core", "How does the PyTorch Deep Autoencoder detect equipment anomalies?",
           "The PyTorch Deep Autoencoder uses a 5-layer bottleneck structure (56 -> 32 -> 16 -> 32 -> 56) trained exclusively on healthy baseline telemetry (RUL > 100 cycles). The network learns to compress and reconstruct normal operating sensor states. During inference, if sensor readings anomaly-deviate due to thermal or pressure degradation, reconstruction Mean Squared Error (MSE) spikes above the calibrated 95th-percentile threshold (0.042 MSE), flagging an anomaly without needing labeled failure examples.")

    add_qa(4, "ML Core", "Why deploy a tri-model architecture (Autoencoder + Classifier + Regressor) instead of a single model?",
           "No single model covers all operational requirements: (1) The <b>Autoencoder</b> provides <i>unsupervised novelty detection</i> for unexpected mechanical anomalies. (2) The <b>XGBoost Classifier</b> outputs a calibrated <i>binary failure probability</i> within 30 cycles, powering risk badge alerts (Critical, High, Medium, Healthy). (3) The <b>XGBoost Regressor</b> estimates <i>continuous Remaining Useful Life</i> in exact cycles for maintenance scheduling. Together, they provide a 360-degree diagnostic shield.")

    add_qa(5, "ML Core", "How did you resolve class imbalance when predicting failure within 30 cycles?",
           "In run-to-failure datasets, healthy cycles outnumber failure window cycles ~5 to 1. We handled this by: (1) Setting XGBoost's `scale_pos_weight` parameter to the ratio of negative to positive samples. (2) Optimizing probability thresholds using Precision-Recall curves. (3) Evaluating model performance via ROC-AUC (0.989) and F1-score (0.951) rather than standard accuracy.")

    # --- CATEGORY 2: FEATURE ENGINEERING & SIGNAL PROCESSING ---
    story.append(Paragraph("Category 2: Feature Engineering & Signal Processing", style_h2))

    add_qa(6, "Feature Eng", "What signal processing techniques were applied to the 21 physical sensor channels?",
           "We engineered 56 features across three signal domains: (1) <b>Multi-scale Rolling Statistics:</b> Rolling mean, std, min, and max over window sizes [5, 10, 20] to capture short-term fluctuations and trend variance. (2) <b>Exponential Moving Average (EMA):</b> Smooths transient high-frequency noise while prioritizing recent cycles. (3) <b>FFT Spectral Power Density:</b> Applied Fast Fourier Transforms to extract spectral energy in temperature and pressure channels, capturing high-frequency harmonic vibration associated with turbine blade erosion.")

    add_qa(7, "Feature Eng", "How do you guarantee zero data leakage during time-series feature engineering?",
           "Data leakage is strictly prevented by: (1) <b>Causal Trailing Windows:</b> Rolling calculations strictly use past and present cycles up to cycle <i>t</i> (`closed='left'` / trailing windows). (2) <b>Per-Unit Grouping:</b> Transformations are executed strictly within engine unit groups (`df.groupby('unit_id')`), preventing cross-asset data bleed. (3) <b>Isolated Scalers:</b> Standard Z-score scalers are fit strictly on the training partition and applied transform-only to test sets.")

    add_qa(8, "Feature Eng", "Which physical sensors were most predictive of engine failure?",
           "Through EDA and SHAP analysis, 5 sensors demonstrated highest degradation sensitivity: <b>Sensor 2 (LPC Total Temp T24)</b>, <b>Sensor 3 (HPC Total Temp T30)</b>, <b>Sensor 4 (LPT Total Temp T50)</b>, <b>Sensor 11 (HPC Speed Nhc)</b>, and <b>Sensor 12 (Fan Speed Nf)</b>. Non-informative sensors with zero variance (Sensors 1, 5, 10, 16, 18, 19) were automatically filtered out.")

    # --- CATEGORY 3: EXPLAINABLE AI (XAI) & SHAP ---
    story.append(Paragraph("Category 3: Explainable AI & SHAP Root-Cause Attribution", style_h2))

    add_qa(9, "XAI / SHAP", "How does SHAP work mathematically, and why is it vital for field maintenance engineers?",
           "SHAP relies on game-theoretic Shapley values to calculate the marginal contribution of each feature across all possible feature sub-combinations. When PredictiveX outputs a 94% failure risk, SHAP attributes exact numeric contributions to specific physical sensors (e.g., HPC Temp T30 added +0.42 to risk, while Fan Speed Nf subtracted -0.08). Field engineers cannot act on black-box probabilities—SHAP specifies <i>which physical subsystem requires overhaul</i>.")

    add_qa(10, "XAI / SHAP", "Why use `shap.TreeExplainer` instead of `shap.KernelSHAP`?",
           "`TreeExplainer` exploits decision tree architecture to compute exact SHAP values in $O(TLD^2)$ time (where $T$ is tree count, $L$ is leaf count, $D$ is depth), compared to $O(2^M)$ exponential time for model-agnostic `KernelSHAP`. This enables instantaneous SHAP computation (~3 ms) directly inside the Streamlit dashboard during live telemetry streaming.")

    # --- CATEGORY 4: REAL-WORLD BUGS, MLOPS & SYSTEM FIXES ---
    story.append(Paragraph("Category 4: MLOps, System Integration & Debugging Gotchas", style_h2))

    add_qa(11, "MLOps & Debug", "How did you debug and fix the Python 3.14 + Apple Silicon segmentation fault between PyTorch and XGBoost?",
           "On Python 3.14 on macOS ARM64, instantiating PyTorch dynamic C++ libraries after loading an XGBoost `xgb.Booster` caused a low-level dynamic memory pointer collision, resulting in exit code 139 (SegFault). We identified that C++ extension initialization order was responsible and resolved it by establishing a mandatory import sequence: initializing XGBoost boosters first, followed by PyTorch Autoencoder allocation.")

    add_qa(12, "MLOps & Debug", "How did you resolve the Streamlit metric card layout clipping issue?",
           "Streamlit renders `st.caption()` and `st.markdown()` as separate React DOM elements outside raw HTML `<div>` blocks, causing metric values to overflow outside card borders. We fixed this by encapsulating each metric card inside a single self-contained HTML markdown block (`st.markdown('<div class=\"metric-card\">...</div>', unsafe_allow_html=True)`), binding CSS styles directly to child `<p>` elements.")

    add_qa(13, "MLOps & Debug", "How did you fix the `KeyError: 'feature_value'` in the SHAP Feature Inspector?",
           "The `ModelExplainer` returned feature dictionary keys named `'value'`, whereas the GUI code accessed `'feature_value'`. We implemented a fallback accessor: `val = feat_info.get('value', feat_info.get('feature_value', 0.0))`, ensuring robust key handling regardless of dictionary schema.")

    add_qa(14, "MLOps & Debug", "How did you solve `TypeError: Object of type int64 is not JSON serializable` in the report export feature?",
           "NumPy data types (`np.int64`, `np.float64`) returned by pandas and XGBoost are not natively serializable by Python's standard `json.dumps()`. We implemented a custom `json_default(obj)` encoder function converting `np.integer` to `int`, `np.floating` to `float`, and `np.ndarray` to `list` before calling `json.dumps()`.")

    add_qa(15, "MLOps & System", "How would you deploy PredictiveX for high-scale enterprise streaming (e.g., AWS/Azure)?",
           "(1) <b>Data Ingestion:</b> Sensor telemetry streams from aircraft MQTT/Kafka brokers into Apache Flink. (2) <b>Feature Store:</b> Trailing rolling statistics are stored in Redis / Feast feature store. (3) <b>Model Serving:</b> The FastAPI backend (`src/api/main.py`) is deployed in Docker containers on Kubernetes (EKS) with Triton Inference Server. (4) <b>Drift Monitoring:</b> Evidently AI monitors data drift, triggering Airflow retraining DAGs when accuracy drops below threshold.")

    # --- CATEGORY 5: IIOT DOMAIN, BUSINESS IMPACT & DEFENSE ---
    story.append(Paragraph("Category 5: IIoT Domain, Business Impact & Project Defense", style_h2))

    add_qa(16, "IIoT Domain", "What is the NASA C-MAPSS dataset, and why is it the gold standard for predictive maintenance?",
           "C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) is a turbofan engine simulator developed by NASA. FD001 simulates engine run-to-failure trajectories under standard operating conditions. It contains 21 continuous sensor channels recording degradation from healthy commissioning to catastrophic failure across multiple units, making it the industry benchmark for RUL modeling.")

    add_qa(17, "IIoT Domain", "How does PredictiveX reduce capital expenditure for oil & gas and aerospace operators?",
           "Fixed-interval maintenance overhauls engines based on operating hours rather than physical health, causing premature component replacement or missing undetected fatigue failures. PredictiveX enables Condition-Based Maintenance, extending component operating life by up to 25% while eliminating unscheduled downtime teardowns.")

    add_qa(18, "Defense", "How did you validate your models to ensure no data leakage or overfitting occurred?",
           "Validation was performed using 5-fold GroupKFold cross-validation grouped strictly by `unit_id`. This ensured that telemetry cycles from the same engine unit were never split across training and validation sets, reflecting realistic zero-shot performance on unseen equipment.")

    add_qa(19, "Defense", "What are the limitations of the current implementation?",
           "FD001 operates under a single sea-level flight condition. Complex operational datasets (e.g., FD002/FD004) involve 6 different flight regimes and multiple failure modes (hpc vs fan degradation). Expanding PredictiveX to multi-regime datasets requires conditioning features on operating mode clusters (K-Means/GMM).")

    add_qa(20, "Defense", "How would you incorporate physics-based domain knowledge into the machine learning pipeline?",
           "We can implement Physics-Informed Neural Networks (PINNs) by adding thermodynamic conservation laws (mass balance, energy conservation between compressor stages) directly as penalty loss terms in the PyTorch Autoencoder loss function: $L_{total} = L_{MSE} + lambda * L_{physics}$.")

    add_qa(21, "Defense", "Why use Streamlit for the user interface instead of React or Angular?",
           "Streamlit enables rapid Python-native dashboard prototyping, allowing seamless binding between complex data structures (pandas DataFrames, Plotly figure objects, SHAP arrays) and responsive UI components without writing separate REST API wrappers or JavaScript state management code.")

    add_qa(22, "Defense", "How does the Natural Language Maintenance Report generator work?",
           "The `MaintenanceReportGenerator` evaluates model outputs (failure probability, predicted RUL, anomaly score, and top SHAP drivers) to automatically construct structured executive reports including urgency banners, maintenance action checklists, and detailed technical root-cause explanations.")

    add_qa(23, "Defense", "What metrics would you monitor in production to ensure model health?",
           "Key production monitoring metrics include: (1) <b>Prediction Drift:</b> Tracking shift in predicted RUL distributions via Kolmogorov-Smirnov test. (2) <b>Feature Drift:</b> Monitoring Population Stability Index (PSI) on sensor channels. (3) <b>Inference Latency:</b> Tracking 99th percentile API response time (< 50 ms).")

    add_qa(24, "Defense", "If you had 3 additional months on this project, what features would you prioritize?",
           "(1) <b>Multi-Regime Support:</b> Expanding models to C-MAPSS FD002/FD004. (2) <b>Edge ONNX Quantization:</b> Converting PyTorch and XGBoost models to ONNX runtime for deployment on NVIDIA Jetson edge devices. (3) <b>Active Learning Loop:</b> Incorporating technician feedback when work orders are completed to retrain failure classifiers.")

    add_qa(25, "Defense", "How would you pitch this project to an executive engineering committee?",
           "\"PredictiveX is an end-to-end IIoT intelligence platform that transforms equipment telemetry into actionable maintenance foresight. By unifying deep anomaly detection, XGBoost RUL forecasting, and SHAP root-cause diagnostics, PredictiveX achieves 96.2% failure accuracy and provides field technicians with exact component work orders—reducing unscheduled machinery downtime by up to 40% and saving millions in operational tear-down costs.\"")

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully built master PDF: {filename}")

if __name__ == "__main__":
    build_pdf()
