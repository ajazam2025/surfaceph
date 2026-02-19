import streamlit as st
import numpy as np
import joblib

st.title("Surface pH Predictor")

model = joblib.load("surface_ph_model.joblib")
scaler = joblib.load("surface_ph_scaler.joblib")

month = st.number_input("Time (month)", 0.0, 60.0)
h2s   = st.number_input("H2S (ppm)", 0.0, 100.0)
temp  = st.number_input("Temperature (°C)", 0.0, 50.0)
rh    = st.number_input("Relative Humidity (%)", 0.0, 100.0)

if st.button("Predict"):
    input_data = np.array([[month, h2s, temp, rh]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    st.success(f"Predicted Surface pH: {prediction:.3f}")
