import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Surface pH Predictor",
    layout="wide",
    page_icon="🧪"
)

# ---------------- STYLING ----------------
st.markdown("""
<style>
.title {
    font-size:32px;
    font-weight:700;
}
.card {
    background-color:#f4f6f9;
    padding:25px;
    border-radius:15px;
    text-align:center;
    font-size:24px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">🧪 Surface pH Prediction System</p>', unsafe_allow_html=True)
st.write("Lightweight Neural Network (Deep Learning Model)")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model_and_scaler():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "surface_ph_regressor.keras")
    scaler_path = os.path.join(base_path, "surface_ph_scaler.joblib")

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler

model, scaler = load_model_and_scaler()

# ---------------- INPUT SECTION ----------------
st.markdown("### 🔢 Enter Environmental Parameters")

col1, col2, col3, col4 = st.columns(4)

with col1:
    month = st.slider("Time (month)", 0.0, 60.0, 12.0)

with col2:
    h2s = st.slider("H₂S Concentration (ppm)", 0.0, 100.0, 5.0)

with col3:
    temp = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)

with col4:
    rh = st.slider("Relative Humidity (%)", 0.0, 100.0, 80.0)

st.markdown("---")

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Surface pH"):

    input_data = np.array([[month, h2s, temp, rh]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled).flatten()[0]

    st.markdown(f"""
    <div class="card">
        Predicted Surface pH <br><br>
        <strong>{prediction:.3f}</strong>
    </div>
    """, unsafe_allow_html=True)
