import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="❤️ Heart Disease Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Robust CSS for light background and full content visibility ---
# Added download button styling CSS at the bottom
st.markdown(
    """
    <style>
    html, body, .stApp {
        background-color: #f8f9fb !important;
        color: #22272a !important;
    }
    .stApp, .block-container, .main, .sidebar-content, .css-1d391kg {
        background-color: #f8f9fb !important;
        color: #22272a !important;
    }
    .main-title {
        color: #ff4b4b !important;
        font-size: 2.5rem !important;
        font-weight: bold !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1) !important;
    }
    /* Main text content and sidebar text extra dark */
    .stMarkdown, .stText, .markdown-text-container, p, span,
    div[data-testid="stMarkdownContainer"], .sidebar-content {
        color: #1a1d21 !important;
        font-size: 1.09rem !important;
        font-weight: 500 !important;
    }
    /* Widget labels more visible */
    .stNumberInput label, .stTextInput label, .stSelectbox label {
        color: #2e86ab !important;
        font-weight: 700 !important;
        font-size: 15px !important;
    }
    /* Inputs and selectors looks bright and text dark */
    .stNumberInput, .stTextInput, .stSelectbox, .stExpander, .stButton, .stTabs {
        background-color: #fff !important;
        color: #1a1d21 !important;
        border-radius: 8px !important;
    }
    .stNumberInput input, .stTextInput input, .stSelectbox select {
        background-color: #fff !important;
        color: #22272a !important;
        font-size: 1.08rem !important;
        font-weight: 600 !important;
        border: 2px solid #e6e6e6 !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }
    .stButton button {
        background-color: #ff4b4b !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        font-size: 1.13rem !important;
        padding: 12px 24px !important;
        transition: all 0.3s ease !important;
    }
    .stButton button:hover {
        background-color: #e04343 !important;
        transform: translateY(-2px) !important;
    }
    /* Dataframe and table */
    .dataframe, .stDataFrame, .css-1l269bu, .stTable {
        background-color: #fff !important;
        color: #22272a !important;
        border-radius: 8px;
        border: 2px solid #e6e6e6;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    /* Expander text and headers */
    .stExpander {
        background-color: #fff !important;
        color: #22272a !important;
        border-radius: 10px !important;
        border: 2px solid #e6e6e6;
    }
    .stExpander .streamlit-expanderHeader {
        font-weight: bold !important;
        color: #22272a !important;
        font-size: 1.12rem !important;
        background-color: #f8f9fa !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }
    .subheader-custom {
        color: #2e86ab !important;
        border-bottom: 3px solid #2e86ab;
        padding-bottom: 12px;
        margin-bottom: 25px;
        font-size: 1.5rem !important;
        font-weight: bold !important;
    }
    [data-testid="metric-container"] {
        background-color: #fff !important;
        color: #101415 !important;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #e6e6e6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #fff;
        border-radius: 10px 10px 0px 0px;
        gap: 8px;
        padding: 10px 20px;
        border: 2px solid #e6e6e6;
        color: #2e86ab;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b !important;
        color: #fff !important;
        border-color: #ff4b4b !important;
    }
    /* Side bar "folder"/file cards */
    .file-list {
        background-color: #f8f9fa !important;
        border: 2px solid #e9ecef !important;
        border-radius: 10px !important;
        padding: 15px !important;
        margin: 10px 0 !important;
    }
    .file-item {
        background-color: #fff !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
        margin: 5px 0 !important;
        color: #495057 !important;
        font-family: 'Courier New', monospace !important;
        font-size: 0.9rem !important;
    }
    .model-file { background-color: #d4edda !important; border-color: #c3e6cb !important; color: #155724 !important; font-weight: 600 !important; }
    .scaler-file { background-color: #d1ecf1 !important; border-color: #bee5eb !important; color: #0c5460 !important; font-weight: 600 !important; }
    .data-file { background-color: #fff3cd !important; border-color: #ffeaa7 !important; color: #856404 !important; font-weight: 600 !important; }
    /* Status indicators */
    .status-success { color: #28a745 !important; font-weight: bold !important; }
    .status-warning { color: #ffc107 !important; font-weight: bold !important; }
    .status-error { color: #dc3545 !important; font-weight: bold !important; }
    /* Info/alert */
    .stAlert, .stInfo, [role="alert"] {
        background-color: #e8f4fd !important;
        border: 2px solid #2e86ab !important;
        border-radius: 10px !important;
        color: #22272a !important;
        font-size: 1.1rem !important;
    }
    /* Custom style for Streamlit download buttons */
    div.stDownloadButton > button {
        background-color: #2e86ab !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding: 10px 25px !important;
        border-radius: 10px !important;
        border: none !important;
        transition: background-color 0.3s ease;
    }
    div.stDownloadButton > button:hover {
        background-color: #1b5f83 !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.markdown('<h1 class="main-title">❤️ Heart Disease Predictor</h1>', unsafe_allow_html=True)

BASE = Path(__file__).parent
DEPLOY_DIR = BASE / "deployed_model"

# -------------- Model/Scaler/Feature loading -------------
st.sidebar.header("📁 Model Folder")
st.sidebar.write(f"`{str(DEPLOY_DIR)}`")

models = {}
scaler = None
feature_names = None
loaded_files = []

if DEPLOY_DIR.exists():
    st.sidebar.markdown('<div class="file-list">', unsafe_allow_html=True)
    st.sidebar.subheader("📋 Available Files")
    for p in DEPLOY_DIR.glob("*"):
        file_type = ""
        if p.suffix == '.pkl':
            if 'scaler' in p.stem.lower():
                file_type = 'scaler-file'
            else:
                file_type = 'model-file'
                try:
                    obj = joblib.load(p)
                    models[p.stem] = obj
                    loaded_files.append(f"✅ {p.name}")
                except Exception as e:
                    loaded_files.append(f"❌ {p.name} (Error: {str(e)[:30]}...)")
        elif p.suffix == '.json':
            file_type = 'data-file'
            loaded_files.append(f"📄 {p.name}")
        else:
            loaded_files.append(f"📁 {p.name}")

    for file_item in loaded_files:
        file_class = "file-item"
        if "✅" in file_item or ".pkl" in file_item and "scaler" not in file_item.lower():
            file_class += " model-file"
        elif "scaler" in file_item.lower():
            file_class += " scaler-file"
        elif ".json" in file_item or ".csv" in file_item:
            file_class += " data-file"
        st.sidebar.markdown(f'<div class="{file_class}">{file_item}</div>', unsafe_allow_html=True)

    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    if (DEPLOY_DIR / "scaler.pkl").exists():
        try:
            scaler = joblib.load(DEPLOY_DIR / "scaler.pkl")
            st.sidebar.markdown('<p class="status-success">✅ Scaler loaded successfully!</p>', unsafe_allow_html=True)
        except Exception as e:
            st.sidebar.markdown(f'<p class="status-error">❌ Scaler load error: {str(e)[:50]}...</p>', unsafe_allow_html=True)
    if (DEPLOY_DIR / "feature_names.json").exists():
        try:
            feature_names = json.load(open(DEPLOY_DIR / "feature_names.json"))
            st.sidebar.markdown('<p class="status-success">✅ Feature names loaded</p>', unsafe_allow_html=True)
        except Exception as e:
            st.sidebar.markdown(f'<p class="status-warning">⚠️ Could not read feature_names.json: {str(e)[:50]}...</p>', unsafe_allow_html=True)

if feature_names is None:
    dataset_path = BASE / "heart_disease_dataset.csv"
    if dataset_path.exists():
        df = pd.read_csv(dataset_path)
        if "heart_disease" in df.columns:
            feature_names = [c for c in df.columns if c != "heart_disease"]
        else:
            feature_names = list(df.columns)
        st.sidebar.markdown('<p class="status-success">✅ Feature names inferred from heart_disease_dataset.csv</p>', unsafe_allow_html=True)

if feature_names is None:
    st.sidebar.markdown('<p class="status-warning">⚠️ Feature names not found.</p>', unsafe_allow_html=True)
    st.sidebar.warning("Paste comma-separated feature names (training order).")
    pasted = st.sidebar.text_area("Paste feature names", placeholder="age,sex,cp,trestbps,...")
    if pasted:
        feature_names = [s.strip() for s in pasted.split(",") if s.strip()]

if feature_names is None:
    st.error("Feature names not available. Provide feature_names.json or heart_disease_dataset.csv.")
    st.stop()

if models:
    selected_model_file = list(models.keys())[0]
    model = models[selected_model_file]
else:
    st.error("No model files found in deployed_model folder. Please add model files.")
    st.stop()

# -------------- Main Tabs: Patient Data and Prediction History -------------
tab1, tab2 = st.tabs(["👤 Patient Data", "📊 Prediction History"])

with tab1:
    st.markdown('<h2 class="subheader-custom">Enter Patient Data</h2>', unsafe_allow_html=True)
    st.write("Provide values for all features to get a heart disease prediction.")

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Model in use:** {selected_model_file}.pkl")
    with col2:
        st.info(f"**Model Type:** {type(model).__name__}")

    defaults = None
    if scaler is not None and hasattr(scaler, "mean_"):
        try:
            defaults = list(scaler.mean_)
        except Exception:
            defaults = None

    input_vals = []
    num_expanders = (len(feature_names) + 4) // 5
    for exp_idx in range(num_expanders):
        start_idx = exp_idx * 5
        end_idx = min((exp_idx + 1) * 5, len(feature_names))
        expander_title = f"Features {start_idx + 1} to {end_idx}"
        with st.expander(expander_title, expanded=(exp_idx == 0)):
            col1, col2 = st.columns(2)
            for i in range(start_idx, end_idx):
                feat = feature_names[i]
                d = float(defaults[i]) if defaults and i < len(defaults) else 0.0
                if (i - start_idx) % 2 == 0:
                    val = col1.number_input(
                        label=feat,
                        value=d,
                        format="%.4f",
                        key=f"input_{i}",
                        help=f"Enter value for {feat}"
                    )
                else:
                    val = col2.number_input(
                        label=feat,
                        value=d,
                        format="%.4f",
                        key=f"input_{i}",
                        help=f"Enter value for {feat}"
                    )
                input_vals.append(val)

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "🚀 PREDICT HEART DISEASE",
            use_container_width=True,
            type="primary",
            key="predict_btn"
        )

    if predict_button:
        try:
            arr = np.array(input_vals).reshape(1, -1)
            model_class = type(model).__name__
            if model_class in ["LogisticRegression", "SVC", "LogisticRegressionCV"] and scaler is not None:
                try:
                    arr = scaler.transform(arr)
                except Exception as e:
                    st.error("Scaling failed: " + str(e))
                    st.stop()
            pred = model.predict(arr)[0]
            prob = None
            if hasattr(model, "predict_proba"):
                try:
                    prob = model.predict_proba(arr)[0, 1]
                except Exception:
                    prob = None

            st.markdown('<h2 class="subheader-custom">🎯 Prediction Results</h2>', unsafe_allow_html=True)
            if int(pred) == 1:
                st.error("##  HEART DISEASE DETECTED")
                if prob is not None:
                    st.metric("Confidence Level", f"{prob:.2%}")
            else:
                st.success("##  NO HEART DISEASE DETECTED")
                if prob is not None:
                    st.metric("Confidence Level", f"{prob:.2%}")

            if prob is not None:
                st.write("Probability of heart disease:")
                st.progress(float(prob))

            st.markdown('<h3 class="subheader-custom">📋 Input Features Used</h3>', unsafe_allow_html=True)
            debug_df = pd.DataFrame({
                "Feature": feature_names,
                "Value": input_vals,
                "Index": range(1, len(feature_names) + 1)
            })
            st.dataframe(debug_df[["Index", "Feature", "Value"]], use_container_width=True)

            st.markdown('<h3 class="subheader-custom">📊 Feature Values Visualization</h3>', unsafe_allow_html=True)
            chart_df = debug_df.set_index("Feature")[["Value"]]
            st.bar_chart(chart_df)

            log_path = BASE / "prediction_log.csv"
            row = {**{f: v for f, v in zip(feature_names, input_vals)},
                   "prediction": int(pred),
                   "probability": float(prob) if prob is not None else np.nan}
            log_df = pd.DataFrame([row])
            columns = feature_names + ["prediction", "probability"]
            log_df.to_csv(
                log_path,
                mode="a",
                header=not log_path.exists(),
                index=False,
                columns=columns
            )
            st.success(f"✅ Prediction saved to log file: {log_path.name}")

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

with tab2:
    st.markdown('<h2 class="subheader-custom">📈 Prediction History</h2>', unsafe_allow_html=True)
    log_path = BASE / "prediction_log.csv"
    if log_path.exists():
        try:
            history = pd.read_csv(log_path)
        except pd.errors.ParserError:
            history = pd.read_csv(log_path, on_bad_lines="skip")

        if not history.empty:
            st.markdown("---")
            with st.expander("🔍 View Last 10 Predictions", expanded=True):
                st.write("**Recent Prediction History:**")
                display_data = history.tail(10).copy()
                display_data = display_data.iloc[::-1]  # Latest first
                display_data['Status'] = display_data['prediction'].apply(
                    lambda x: 'Heart Disease' if x == 1 else 'No Heart Disease'
                )
                if 'probability' in display_data.columns:
                    display_data['Probability'] = display_data['probability'].apply(
                        lambda x: f"{x:.2%}" if not pd.isna(x) else "N/A"
                    )
                display_cols = ['Status']
                if 'probability' in display_data.columns:
                    display_cols.append('Probability')
                for feat in feature_names[:3]:
                    if feat in display_data.columns:
                        display_cols.append(feat)
                st.dataframe(display_data[display_cols], use_container_width=True)

            st.markdown("---")
            with st.expander("📋 View Complete Prediction Data"):
                st.write("**Complete Prediction History (All Features):**")
                st.dataframe(history, use_container_width=True)

            st.markdown("---")
            st.write("### 💾 Data Export")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download Complete Log (CSV)",
                    data=history.to_csv(index=False),
                    file_name="heart_disease_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col2:
                st.download_button(
                    "Download Last 10 Predictions (CSV)",
                    data=history.tail(10).to_csv(index=False),
                    file_name="recent_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.info("📭 Prediction log exists but is empty. Make your first prediction in the 'Patient Data' tab!")
            st.markdown("""
            **To get started:**
            1. Go to the **'Patient Data'** tab
            2. Enter patient feature values
            3. Click **'PREDICT HEART DISEASE'**
            4. Your prediction will appear here!
            """)
    else:
        st.info("📭 No prediction history available yet. Make your first prediction to see it here!")
        st.markdown("""
        **To create your first prediction:**
        1. Switch to the **'Patient Data'** tab above
        2. Fill in the patient feature values
        3. Press the **'PREDICT HEART DISEASE'** button
        4. Come back here to see your prediction history!
        """)
        if st.button("🚀 Go to Patient Data Tab to Make First Prediction", use_container_width=True):
            st.session_state.active_tab = "👤 Patient Data"
            st.rerun()

# -------- Sidebar status indicators -----------
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI Model Status")
if models:
    st.sidebar.markdown('<p class="status-success">✅ AI Model loaded successfully!</p>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<p class="status-error">❌ AI Model not loaded!</p>', unsafe_allow_html=True)
if scaler is not None:
    st.sidebar.markdown('<p class="status-success">✅ Data scalar loaded successfully!</p>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<p class="status-warning">⚠️ Data scalar not loaded</p>', unsafe_allow_html=True)
if feature_names is not None:
    st.sidebar.markdown('<p class="status-success">✅ Feature names loaded</p>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<p class="status-error">❌ Feature names not available</p>', unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Quick Info")
st.sidebar.write(f"**Total features:** {len(feature_names)}")
st.sidebar.write(f"**Model in use:** {selected_model_file}")
log_path = BASE / "prediction_log.csv"
if log_path.exists():
    try:
        history = pd.read_csv(log_path)
        st.sidebar.write(f"**Predictions made:** {len(history)}")
    except:
        st.sidebar.write("**Predictions made:** 0")
else:
    st.sidebar.write("**Predictions made:** 0")
