import streamlit as st
import numpy as np
import joblib
import os

# ===================== LOAD MODEL (SAFE PATH) =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "../models/model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "../models/scaler.pkl"))

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)

# ===================== CUSTOM CSS =====================
st.markdown("""
<style>
body {
    background-color: #020617;
}
h1, h2, h3 {
    color: #38bdf8;
}
.stButton>button {
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #0f172a);
}
</style>
""", unsafe_allow_html=True)

# ===================== HEADER =====================
st.markdown("""
<h1 style='text-align: center;'>⚙️ Predictive Maintenance Dashboard</h1>
<p style='text-align: center;'>AI-powered Machine Failure Prediction System</p>
<hr>
""", unsafe_allow_html=True)

# ===================== SIDEBAR =====================
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("### Adjust Machine Parameters")

# All inputs as sliders
air_temp = st.sidebar.slider("Air Temperature (K)", 250.0, 350.0, 300.0)
process_temp = st.sidebar.slider("Process Temperature (K)", 250.0, 350.0, 310.0)
rpm = st.sidebar.slider("Rotational Speed (RPM)", 1000, 3000, 1500)
torque = st.sidebar.slider("Torque (Nm)", 0.0, 100.0, 40.0)
tool_wear = st.sidebar.slider("Tool Wear (min)", 0, 300, 10)

machine_type = st.sidebar.selectbox("Machine Type", ["L", "M"])
type_M = 1 if machine_type == "M" else 0

predict_btn = st.sidebar.button("🚀 Predict Status")

# ===================== MAIN CONTENT =====================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Machine Status")

    if predict_btn:

        input_data = np.array([[ 
            air_temp,
            process_temp,
            rpm,
            torque,
            tool_wear,
            type_M
        ]])

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.error(f"⚠️ High Risk of Machine Failure\n\nConfidence: {probability:.2f}")
        else:
            st.success(f"✅ Machine Operating Normally\n\nConfidence: {1 - probability:.2f}")

        # Progress bar
        st.progress(float(probability))

    else:
        st.info("Adjust parameters from sidebar and click Predict 🚀")

# ===================== FEATURE IMPORTANCE =====================
with col2:
    st.subheader("📈 Feature Importance")

    try:
        importance = model.feature_importances_

        features = [
            "Air Temp",
            "Process Temp",
            "RPM",
            "Torque",
            "Tool Wear",
            "Type_M"
        ]

        import pandas as pd

        df_imp = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(df_imp.set_index("Feature"))

    except:
        st.warning("Feature importance not available")

# ===================== FOOTER =====================
st.markdown("---")
st.caption("Developed by Habibullah Salmani 🚀")