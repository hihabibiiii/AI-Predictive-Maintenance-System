import streamlit as st
import numpy as np
import joblib

# ===================== LOAD MODEL =====================
model = joblib.load("../models/model.pkl")
scaler = joblib.load("../models/scaler.pkl")


# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="centered"
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
</style>
""", unsafe_allow_html=True)

# ===================== HEADER =====================
st.markdown("""
<h1 style='text-align: center;'>⚙️ Predictive Maintenance</h1>
<p style='text-align: center; font-size:18px;'>
AI-powered Machine Failure Detection System
</p>
<hr>
""", unsafe_allow_html=True)

# ===================== INPUT SECTION =====================
st.markdown("### 🛠️ Enter Machine Parameters")

col1, col2 = st.columns(2)

with col1:
    air_temp = st.number_input("Air Temperature (K)", value=300.0)
    rpm = st.number_input("Rotational Speed (RPM)", value=1500)

with col2:
    process_temp = st.number_input("Process Temperature (K)", value=310.0)
    torque = st.number_input("Torque (Nm)", value=40.0)

tool_wear = st.slider("Tool Wear (min)", 0, 300, 10)

# 🔥 IMPORTANT FIX (6th feature)
machine_type = st.selectbox("Machine Type", ["L", "M"])
type_M = 1 if machine_type == "M" else 0

st.markdown("---")

# ===================== PREDICTION =====================
if st.button("🚀 Predict Machine Status"):

    # Correct feature order
    input_data = np.array([[ 
        air_temp,
        process_temp,
        rpm,
        torque,
        tool_wear,
        type_M
    ]])

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("## 📊 Prediction Result")

    if prediction == 1:
        st.markdown(f"""
        <div style="padding:20px; border-radius:10px; background-color:#7f1d1d;">
            <h3>⚠️ High Risk of Machine Failure</h3>
            <p>Confidence: {probability:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="padding:20px; border-radius:10px; background-color:#064e3b;">
            <h3>✅ Machine Operating Normally</h3>
            <p>Confidence: {1 - probability:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

# ===================== FEATURE IMPORTANCE =====================
if st.checkbox("📈 Show Feature Importance"):
    import pandas as pd

    importance = model.feature_importances_

    features = [
        "Air Temp",
        "Process Temp",
        "RPM",
        "Torque",
        "Tool Wear",
        "Type_M"
    ]

    df_imp = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(df_imp.set_index("Feature"))

# ===================== FOOTER =====================
st.markdown("---")
st.caption("Developed by Habibullah Salmani 🚀")