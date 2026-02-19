import streamlit as st
import numpy as np
import joblib
import os

# -------- PAGE CONFIG --------
st.set_page_config(page_title="Surface pH Predictor", layout="centered")

st.title("🧪 Surface pH Predictor")
st.write("Enter environmental conditions to predict surface pH.")

# -------- LOAD MODEL --------
@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    model = joblib.load(os.path.join(base_path, "surface_ph_model.joblib"))
    scaler = joblib.load(os.path.join(base_path, "surface_ph_scaler.joblib"))
    return model, scaler

model, scaler = load_model()

# -------- INPUTS --------
month = st.number_input("Time (month)", 0.0, 60.0, 12.0)
h2s = st.number_input("H2S (ppm)", 0.0, 100.0, 5.0)
temp = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
rh = st.number_input("Relative Humidity (%)", 0.0, 100.0, 80.0)

# -------- PREDICTION --------
if st.button("Predict"):

    input_data = np.array([[month, h2s, temp, rh]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.success(f"Predicted Surface pH: {prediction:.3f}")

