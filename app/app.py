import streamlit as st
import numpy as np
import joblib

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Predictive Maintenance", layout="wide")

# ==============================
# STYLE (Cyber Dark UI 😎)
# ==============================
st.markdown("""
<style>
.stApp {
    background: #0e1117;
    color: white;
}
h1, h2, h3 {
    color: #00f5ff;
}
.stButton>button {
    background-color: #00f5ff;
    color: black;
    border-radius: 10px;
    padding: 10px 20px;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# LOAD MODEL
# ==============================
model = joblib.load("../models/model.pkl")
scaler = joblib.load("../models/scaler.pkl")

# ==============================
# TITLE
# ==============================
st.title("🔧 AI Predictive Maintenance Dashboard")

# ==============================
# SIDEBAR INPUT
# ==============================
st.sidebar.header("⚙️ Machine Inputs")

air_temp = st.sidebar.slider("Air Temperature (K)", 290.0, 320.0, 300.0)
process_temp = st.sidebar.slider("Process Temperature (K)", 300.0, 350.0, 310.0)
rpm = st.sidebar.slider("Rotational Speed (rpm)", 1000, 2000, 1500)
torque = st.sidebar.slider("Torque (Nm)", 10.0, 100.0, 40.0)
tool_wear = st.sidebar.slider("Tool Wear (min)", 0, 300, 10)

type_option = st.sidebar.selectbox("Machine Type", ["Low Quality", "Medium Quality", "High Quality"])

# Encoding
type_L = 1 if type_option == "L" else 0
type_M = 1 if type_option == "M" else 0

# ==============================
# MAIN DISPLAY
# ==============================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Input Summary")
    st.write(f"Air Temp: {air_temp}")
    st.write(f"Process Temp: {process_temp}")
    st.write(f"RPM: {rpm}")
    st.write(f"Torque: {torque}")
    st.write(f"Tool Wear: {tool_wear}")
    st.write(f"Type: {type_option}")

with col2:
    st.subheader("🔍 Prediction")

    if st.button("Predict"):

        input_data = np.array([[air_temp, process_temp, rpm, torque, tool_wear, type_L, type_M]])
        input_scaled = scaler.transform(input_data)

        prob = model.predict_proba(input_scaled)[0][1]
        pred = 1 if prob > 0.3 else 0

        if pred == 1:
            st.error(f"⚠️ High Failure Risk ({prob:.2%})")
        else:
            st.success(f"✅ Machine is Safe ({prob:.2%})")

        # Progress bar
        st.progress(float(prob))

        # Risk label
        if prob < 0.3:
            st.info("🟢 Low Risk")
        elif prob < 0.7:
            st.warning("🟡 Medium Risk")
        else:
            st.error("🔴 High Risk")