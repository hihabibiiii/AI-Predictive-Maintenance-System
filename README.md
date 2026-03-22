# 🔧 AI Predictive Maintenance System

This project predicts machine failures using Machine Learning (XGBoost) and provides a real-time dashboard using Streamlit.

---

## 🚀 Features

- Predict machine failure probability
- Real-time interactive dashboard
- High recall model (safety-focused)
- Industrial use-case implementation

---

## 🧠 Model Details

- Algorithm: XGBoost
- ROC-AUC: ~0.97
- Recall (Failure): ~0.96
- Optimized for minimizing missed failures

---

## 📊 Input Features

- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear
- Machine Type (L, M, H)

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py