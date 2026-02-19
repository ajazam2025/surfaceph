import streamlit as st
import numpy as np
import joblib
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Surface pH Predictor", layout="centered")

st.title("🧪 Surface pH Prediction System")
st.write("Predict surface pH based on environmental conditions.")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    model = joblib.load(os.path.join(base_path, "surface_ph_model.joblib"))
    scaler = joblib.load(os.path.join(base_path, "surface_ph_scaler.joblib"))
    return model, scaler

model, scaler = load_model()

# ---------------- INPUT SECTION ----------------
st.markdown("### Enter Parameters")

month = st.number_input("Time (month)", min_value=0.0, max_value=60.0, value=12.0)
h2s   = st.number_input("H₂S Concentration (ppm)", min_value=0.0, max_value=100.0, value=5.0)
temp  = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
rh    = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=80.0)

# ---------------- PREDICT ----------------
if st.button("🚀 Predict Surface pH"):

    input_data = np.array([[month, h2s, temp, rh]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.success(f"Predicted Surface pH: {prediction:.3f}")
