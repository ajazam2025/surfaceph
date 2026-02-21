import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="GUI for Surface pH Predictor",
    layout="centered",
    page_icon="🧪"
)

# ================= MODERN CSS =================
st.markdown("""
<style>

/* spacing */
.block-container {
    padding-top: 2rem !important;
}

/* background */
.stApp {
    background: linear-gradient(135deg,#f5f7fa,#e4ecf7);
}

/* title */
.main-title {
    font-size: 32px;
    font-weight: 900;
    text-align: center;
    color: #1f2a44;
}

/* section header */
.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 10px;
}

/* prediction cards */
.pred-box {
    border-radius: 14px;
    padding: 16px;
    text-align: center;
    color: white;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 10px;
    box-shadow: 0 5px 14px rgba(0,0,0,0.12);
}

/* footer */
.footer-text {
    text-align: center;
    color: #666;
    font-size: 12px;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown('<div class="main-title">🧪 GUI for Surface pH Predictor</div>', unsafe_allow_html=True)

# ================= TRAIN MODELS =================
@st.cache_resource
def train_models():

    df = pd.read_csv("surface_ph_data.csv")

    X = df[[
        "Time (month)",
        "H2S Concentration (ppm)",
        "Temperature (C)",
        "Relative Humidity (%)"
    ]]
    y = df["Surface PH"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, _, y_train, _ = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    models = {
        "SVR": SVR(),
        "DT": DecisionTreeRegressor(),
        "RF": RandomForestRegressor(n_estimators=50, random_state=42),
        "ADB": AdaBoostRegressor(n_estimators=50, random_state=42),
        "MLP": MLPRegressor(hidden_layer_sizes=(32,16), max_iter=500, random_state=42),
        "LNN": MLPRegressor(hidden_layer_sizes=(16,), max_iter=500, random_state=42),
    }

    for m in models.values():
        m.fit(X_train, y_train)

    return models, scaler

models, scaler = train_models()

# ================= INPUT SECTION =================
st.markdown('<div class="section-title">🔷 Input Parameters</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    month = st.number_input("🕒 Time (month)", value=12.0, step=0.1)
    temp  = st.number_input("🌡 Temperature (°C)", value=25.0, step=0.1)

with col2:
    h2s = st.number_input("🧪 H₂S (ppm)", value=5.0, step=0.1)
    rh  = st.number_input("💧 Humidity (%)", value=80.0, step=0.1)

st.markdown("")

predict_clicked = st.button("🚀 Predict Using All Models", use_container_width=True)

# ================= PREDICTIONS =================
if predict_clicked:

    X_new = np.array([[month, h2s, temp, rh]])
    X_new_scaled = scaler.transform(X_new)

    preds = {name: model.predict(X_new_scaled)[0] for name, model in models.items()}

    st.markdown("### 📊 Model Predictions")

    colors = [
        "#4e73df", "#1cc88a", "#36b9cc",
        "#f6c23e", "#e74a3b", "#6f42c1"
    ]

    colA, colB = st.columns(2)
    items = list(preds.items())

    for i, (name, value) in enumerate(items):
        target_col = colA if i % 2 == 0 else colB
        with target_col:
            st.markdown(
                f"""
                <div class="pred-box" style="background:{colors[i]};">
                    {name}<br>{value:.3f}
                </div>
                """,
                unsafe_allow_html=True
            )

# ================= FOOTER =================
st.markdown(
    '<div class="footer-text">Developed by <b>Tasaduq Ismail Wani</b> • BITS Pilani</div>',
    unsafe_allow_html=True
)
