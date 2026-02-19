import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Surface pH Predictor")

st.title("Surface pH Prediction")

# ---------------- DEBUG FILE CHECK ----------------
base_path = os.path.dirname(__file__)
files_here = os.listdir(base_path)
st.write("Files in directory:", files_here)

# ---------------- LOAD MODEL ----------------
try:
    model = joblib.load(os.path.join(base_path, "surface_ph_model.joblib"))
    scaler = joblib.load(os.path.join(base_path, "surface_ph_scaler.joblib"))
except Exception as e:
    st.error("Model loading failed.")
    st.error(str(e))
    st.stop()


# ---------------- PREDICT ----------------
if st.button("Predict"):

    try:
        input_data = np.array([[month, h2s, temp, rh]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        st.success(f"Predicted Surface pH: {prediction:.3f}")
    except Exception as e:
        st.error("Prediction failed.")
        st.error(str(e))
