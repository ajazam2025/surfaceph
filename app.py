import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Surface pH Predictor",
    layout="wide",
    page_icon="🧪"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.big-font {
    font-size:28px !important;
    font-weight: bold;
}
.metric-box {
    background-color: #f4f6f9;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 22px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">🧪 Surface pH Prediction Dashboard</p>', unsafe_allow_html=True)
st.write("Predict surface pH based on environmental exposure parameters.")

# ---------------- LOAD DATA & TRAIN ONCE ----------------
@st.cache_resource
def load_and_train():

    df = pd.read_csv("surface_ph_data.csv")

    X = df[['Time (month)',
            'H2S Concentration (ppm)',
            'Temperature (C)',
            'Relative Humidity (%)']]

    y = df['Surface PH']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

    models = {}

    models['SVR'] = SVR().fit(X_train, y_train)
    models['DT'] = DecisionTreeRegressor().fit(X_train, y_train)
    models['RF'] = RandomForestRegressor(n_estimators=50).fit(X_train, y_train)
    models['ADB'] = AdaBoostRegressor(n_estimators=50).fit(X_train, y_train)
    models['MLP'] = MLPRegressor(max_iter=500).fit(X_train, y_train)

    # Lightweight Neural Network
    lnn = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(1)
    ])
    lnn.compile(optimizer='adam', loss='mse')
    lnn.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

    models['LNN'] = lnn

    return models, scaler

models, scaler = load_and_train()

# ---------------- INPUT SECTION ----------------
st.markdown("### 🔢 Input Parameters")

col1, col2, col3, col4 = st.columns(4)

with col1:
    month = st.slider("Time (month)", 0.0, 60.0, 10.0)

with col2:
    h2s = st.slider("H2S (ppm)", 0.0, 100.0, 5.0)

with col3:
    temp = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)

with col4:
    rh = st.slider("Relative Humidity (%)", 0.0, 100.0, 80.0)

st.markdown("---")

# ---------------- MODEL SELECTION ----------------
selected_model = st.selectbox(
    "Select Prediction Model",
    ["SVR", "DT", "RF", "ADB", "MLP", "LNN"]
)

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Surface pH"):

    input_data = np.array([[month, h2s, temp, rh]])
    input_scaled = scaler.transform(input_data)

    if selected_model == "LNN":
        prediction = models['LNN'].predict(input_scaled).flatten()[0]
    else:
        prediction = models[selected_model].predict(input_scaled)[0]

    st.markdown(f"""
    <div class="metric-box">
        Predicted Surface pH <br><br>
        <strong>{prediction:.3f}</strong>
    </div>
    """, unsafe_allow_html=True)
