import streamlit as st, pandas as pd, joblib, json
from pathlib import Path
import yaml

st.set_page_config(page_title="Cancer Risk (Demo)", page_icon="🩺", layout="centered")
st.title("🩺 Cancer Risk Prediction (Demo)")
st.caption("Educational demo — not medical advice.")

# Load model + threshold
model_path = Path("../models/artifacts/best_pipeline.joblib")
metrics_path = Path("../models/metrics/val_metrics.json")
threshold = 0.50
if metrics_path.exists():
    threshold = json.loads(metrics_path.read_text())["threshold"]

if not model_path.exists():
    st.error("Model not found. Run training first (scripts/run_baselines.py).")
    st.stop()

pipe = joblib.load(model_path)

# UI
age = st.slider("Age", 20, 80, 45)
gender = st.selectbox("Gender", [0,1], format_func=lambda x: "Female" if x==1 else "Male")
bmi = st.slider("BMI", 15.0, 40.0, 24.0, 0.1)
smoking = st.selectbox("Smoking", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
genetic = st.selectbox("Genetic Risk", [0,1,2], format_func=lambda x: ["Low","Medium","High"][x])
pa = st.slider("Physical Activity (hrs/week)", 0.0, 10.0, 3.0, 0.5)
alcohol = st.slider("Alcohol Intake (units/week)", 0.0, 5.0, 1.0, 0.5)
history = st.selectbox("Personal Cancer History", [0,1], format_func=lambda x: "Yes" if x==1 else "No")

row = {
    "Age": age,
    "Gender": gender,
    "BMI": bmi,
    "Smoking": smoking,
    "GeneticRisk": genetic,
    "PhysicalActivity": pa,
    "AlcoholIntake": alcohol,
    "CancerHistory": history
}
# engineered features
row["Age2"] = row["Age"]**2
row["BMI2"] = row["BMI"]**2
row["BMI_x_PhysicalActivity"] = row["BMI"]*row["PhysicalActivity"]
row["Smoking_x_GeneticRisk"] = row["Smoking"]*row["GeneticRisk"]
row["History_x_GeneticRisk"] = row["CancerHistory"]*row["GeneticRisk"]

X = pd.DataFrame([row])
prob = float(pipe.predict_proba(X)[0,1])
pred = int(prob >= threshold)

st.subheader("Prediction")
st.metric("Estimated probability", f"{prob:.2%}")
st.write(f"Decision (threshold={threshold:.2f}): **{'Cancer' if pred==1 else 'No Cancer'}**")
with st.expander("Inputs"):
    st.json(row)
st.info("This is a demo and not a medical device.")
