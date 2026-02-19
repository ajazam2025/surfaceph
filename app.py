import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Surface pH Predictor", layout="wide")
st.title("Surface pH Prediction Using Machine Learning Models")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("data/surface_ph_data.csv")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------- FEATURES ----------------
X = df[['Time (month)',
        'H2S Concentration (ppm)',
        'Temperature (C)',
        'Relative Humidity (%)']]

y = df['Surface PH']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# ---------------- SKLEARN MODELS ----------------
models = {
    'SVR': SVR(),
    'DT': DecisionTreeRegressor(),
    'RF': RandomForestRegressor(),
    'ADB': AdaBoostRegressor(),
    'MLP': MLPRegressor(max_iter=2000)
}

results = []

st.subheader("Training Models...")

# Train sklearn models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results.append([name, r2, rmse])

# ---------------- LIGHTWEIGHT NEURAL NETWORK ----------------
def build_lnn():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer='adam',
        loss='mse'
    )
    return model

lnn = build_lnn()
lnn.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)

y_pred_lnn = lnn.predict(X_test).flatten()

r2_lnn = r2_score(y_test, y_pred_lnn)
rmse_lnn = np.sqrt(mean_squared_error(y_test, y_pred_lnn))

results.append(["LNN", r2_lnn, rmse_lnn])

results_df = pd.DataFrame(results, columns=["Model", "R2", "RMSE"])

st.success("All Models Trained Successfully")
st.dataframe(results_df)

# ---------------- PERFORMANCE PLOT ----------------
st.subheader("Model Comparison (R2)")

fig, ax = plt.subplots()
ax.bar(results_df["Model"], results_df["R2"])
ax.set_ylabel("R2 Score")
st.pyplot(fig)

# ---------------- PREDICTION ----------------
st.subheader("Make Prediction")

col1, col2 = st.columns(2)

with col1:
    month = st.number_input("Time (month)", 0.0, 60.0)
    h2s = st.number_input("H2S Concentration (ppm)", 0.0, 100.0)

with col2:
    temp = st.number_input("Temperature (C)", 0.0, 50.0)
    rh = st.number_input("Relative Humidity (%)", 0.0, 100.0)

selected_model = st.selectbox(
    "Select Model",
    ["SVR", "DT", "RF", "ADB", "MLP", "LNN"]
)

if st.button("Predict Surface pH"):

    input_data = np.array([[month, h2s, temp, rh]])
    input_scaled = scaler.transform(input_data)

    if selected_model == "LNN":
        prediction = lnn.predict(input_scaled).flatten()[0]
    else:
        model = models[selected_model]
        prediction = model.predict(input_scaled)[0]

    st.success(f"Predicted Surface pH: {prediction:.3f}")

    if selected_model in ["RF", "DT", "ADB"]:
        st.subheader("Feature Importance")
        importance = models[selected_model].feature_importances_
        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importance
        })
        st.bar_chart(importance_df.set_index("Feature"))
