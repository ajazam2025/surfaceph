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
    page_title="Surface pH Predictor",
    layout="centered",
    page_icon="🧪"
)

# ================= COMPACT BEAUTIFUL CSS =================
st.markdown("""
<style>

/* Tight page spacing */
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

/* Background gradient */
.stApp {
    background: linear-gradient(135deg,#f5f7fa,#e4ecf7);
}

/* ⭐ Visible premium title */
.main-title {
    font-size: 32px;
    font-weight: 900;
    text-align: center;
    margin-bottom: 2px;
    color: #1f2a44;
    text-shadow: 0 1px 2px rgba(0,0,0,0.15);
}

/* Subtitle */
.sub-text {
    text-align: center;
    color: #4a5568;
    margin-bottom: 12px;
    font-size: 14px;
}

/* Prediction cards */
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

/* Footer */
.footer-text {
    text-align: center;
    color: #666;
    font-size: 12px;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown('<div class="main-title">🧪 Surface pH Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Multi-Model AI Environmental Assessment</div>', unsafe_allow_html=True)

# ================= TRAIN MODELS (CACHED) =================
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

    X_train, X_test, y_train, y_test = train_test_split(
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

# ================= COMPACT INPUT GRID =================
c1, c2 = st.columns(2)

with c1:
    month = st.slider("Time (month)", 0.0, 60.0, 12.0)
    temp  = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)

with c2:
    h2s = st.slider("H₂S (ppm)", 0.0, 100.0, 5.0)
    rh  = st.slider("Humidity (%)", 0.0, 100.0, 80.0)

st.markdown("")

# ================= PREDICT BUTTON =================
if st.button("🚀 Predict Using All Models", use_container_width=True):

    X_new = np.array([[month, h2s, temp, rh]])
    X_new_scaled = scaler.transform(X_new)

    preds = {name: model.predict(X_new_scaled)[0] for name, model in models.items()}

    colors = [
        "#4e73df", "#1cc88a", "#36b9cc",
        "#f6c23e", "#e74a3b", "#6f42c1"
    ]

    # ⭐ two-column compact grid
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
