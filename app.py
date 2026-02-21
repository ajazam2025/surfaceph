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
    layout="wide",
    page_icon="🧪"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}
.sub-text {
    text-align: center;
    color: gray;
    margin-bottom: 25px;
}
.pred-card {
    background: linear-gradient(135deg,#4e73df,#1cc88a);
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    color: white;
    font-size: 30px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🧪 Surface pH Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Machine Learning Based Environmental Corrosion Assessment</div>', unsafe_allow_html=True)

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

# ================= INPUT SECTION =================
st.markdown("### 🔢 Input Parameters")

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

# ================= MODEL SELECT =================
model_name = st.selectbox(
    "🤖 Select Prediction Model",
    list(models.keys())
)

# ================= PREDICTION =================
if st.button("🚀 Predict Surface pH", use_container_width=True):

    X_new = np.array([[month, h2s, temp, rh]])
    X_new_scaled = scaler.transform(X_new)

    pred = models[model_name].predict(X_new_scaled)[0]

    st.markdown(
        f"""
        <div class="pred-card">
            Predicted Surface pH<br><br>
            {pred:.3f}
        </div>
        """,
        unsafe_allow_html=True
    )

# ================= FOOTER =================
st.markdown("---")
st.caption("Developed for Surface Corrosion Assessment using Machine Learning")
