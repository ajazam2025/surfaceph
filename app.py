import streamlit as st
import numpy as np
import joblib
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Surface pH Predictor", layout="centered")

st.title("🧪 Surface pH Predictor")
st.write("Enter environmental parameters to predict surface pH.")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(__file__)
    model_path = os.path.join(base, "surface_ph_model.joblib")
    scaler_path = os.path.join(base, "surface_ph_scaler.joblib")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_artifacts()

# ---------------- INPUTS ----------------
month = st.number_input("Time (month)", 0.0, 60.0, 12.0)
h2s   = st.number_input("H2S Concentration (ppm)", 0.0, 100.0, 5.0)
temp  = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
rh    = st.number_input("Relative Humidity (%)", 0.0, 100.0, 80.0)

# ---------------- PREDICT ----------------
if st.button("🚀 Predict Surface pH"):
    try:
        X = np.array([[month, h2s, temp, rh]])
        Xs = scaler.transform(X)
        pred = model.predict(Xs)[0]
        st.success(f"Predicted Surface pH: {pred:.3f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
