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

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>
.main-title {
    font-size:36px;
    font-weight:700;
    text-align:center;
    margin-bottom:10px;
}

.sub-title {
    font-size:18px;
    text-align:center;
    color:gray;
    margin-bottom:30px;
}

.prediction-box {
    background: linear-gradient(135deg, #4e73df, #1cc88a);
    padding:30px;
    border-radius:20px;
    text-align:center;
    color:white;
    font-size:28px;
    font-weight:600;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🧪 Surface pH Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Deep Learning Model for Environmental Exposure Analysis</div>', unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "surface_ph_regressor.keras")
    scaler_path = os.path.join(base_path, "surface_ph_scaler.joblib")

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler

model, scaler = load_model()

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

    prediction = model.predict(input_scaled, verbose=0).flatten()[0]

    st.markdown(
        f"""
        <div class="prediction-box">
            Predicted Surface pH<br><br>
            {prediction:.3f}
        </div>
        """,
        unsafe_allow_html=True
    )
