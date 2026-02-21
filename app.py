import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Surface pH Predictor",
    layout="wide",
    page_icon="🧪"
)

# ================= BEAUTIFUL CSS =================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg,#f5f7fa,#e4ecf7);
}

.main-title {
    font-size:42px;
    font-weight:800;
    text-align:center;
    margin-bottom:0px;
}

.sub-text {
    text-align:center;
    color:#555;
    margin-bottom:25px;
}

/* Prediction cards */
.pred-box {
    border-radius:18px;
    padding:22px;
    text-align:center;
    color:white;
    font-size:20px;
    font-weight:700;
    box-shadow:0 8px 20px rgba(0,0,0,0.15);
}

.footer-text {
    text-align:center;
    color:#666;
    font-size:14px;
    margin-top:40px;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown('<div class="main-title">🧪 Surface pH Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Multi-Model AI Prediction System</div>', unsafe_allow_html=True)

# ================= TRAIN ONCE =================
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
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
        "AdaBoost": AdaBoostRegressor(n_estimators=50, random_state=42),
        "MLP": MLPRegressor(hidden_layer_sizes=(32,16), max_iter=500, random_state=42),
        "LNN": MLPRegressor(hidden_layer_sizes=(16,), max_iter=500, random_state=42),
    }

    for m in models.values():
        m.fit(X_train, y_train)

    return models, scaler

models, scaler = train_models()

# ================= INPUT PANEL =================
st.markdown("### 🔢 Input Environmental Parameters")

c1, c2, c3, c4 = st.columns(4)

with c1:
    month = st.slider("Time (month)", 0.0, 60.0, 12.0)

with c2:
    h2s = st.slider("H₂S (ppm)", 0.0, 100.0, 5.0)

with c3:
    temp = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)

with c4:
    rh = st.slider("Relative Humidity (%)", 0.0, 100.0, 80.0)

st.markdown("---")

# ================= PREDICT ALL =================
if st.button("🚀 Predict Using All Models", use_container_width=True):

    X_new = np.array([[month, h2s, temp, rh]])
    X_new_scaled = scaler.transform(X_new)

    preds = {name: model.predict(X_new_scaled)[0] for name, model in models.items()}

    # --------- BEAUTIFUL CARDS ----------
    st.markdown("### 📊 Model Predictions")

    colors = [
        "#4e73df", "#1cc88a", "#36b9cc",
        "#f6c23e", "#e74a3b", "#6f42c1"
    ]

    cols = st.columns(3)

    for i, (name, value) in enumerate(preds.items()):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="pred-box" style="background:{colors[i]};">
                    {name}<br><br>
                    {value:.3f}
                </div>
                """,
                unsafe_allow_html=True
            )

    # --------- BAR CHART ----------
    st.markdown("### 📈 Model Comparison")

    fig, ax = plt.subplots()
    ax.bar(preds.keys(), preds.values())
    ax.set_ylabel("Predicted Surface pH")
    ax.set_xticklabels(preds.keys(), rotation=45)
    st.pyplot(fig)

# ================= FOOTER =================
st.markdown(
    '<div class="footer-text">Developed by <b>Tasaduq Ismail Wani</b> • BITS Pilani</div>',
    unsafe_allow_html=True
)
