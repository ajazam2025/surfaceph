import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Surface pH Predictor", layout="wide")
st.title("Surface pH Prediction Using Machine Learning")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "surface_ph_data.csv")
    return pd.read_csv(file_path)

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------- FEATURES ----------------
X = df[['Time (month)',
        'H2S Concentration (ppm)',
        'Temperature (C)',
        'Relative Humidity (%)']]

y = df['Surface PH']

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# ---------------- TRAIN BUTTON ----------------
if st.button("Train Models"):

    st.write("Training... Please wait")

    # -------- Sklearn Models --------
    models = {
        'SVR': SVR(),
        'DT': DecisionTreeRegressor(),
        'RF': RandomForestRegressor(n_estimators=50),
        'ADB': AdaBoostRegressor(n_estimators=50),
        'MLP': MLPRegressor(max_iter=500)
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append([name, r2, rmse])

    # -------- Lightweight Neural Network --------
    lnn = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(1)
    ])

    lnn.compile(optimizer='adam', loss='mse')
    lnn.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

    y_pred_lnn = lnn.predict(X_test).flatten()
    r2_lnn = r2_score(y_test, y_pred_lnn)
    rmse_lnn = np.sqrt(mean_squared_error(y_test, y_pred_lnn))

    results.append(["LNN", r2_lnn, rmse_lnn])

    results_df = pd.DataFrame(results, columns=["Model", "R2", "RMSE"])

    st.success("Training Completed!")
    st.dataframe(results_df)

    # -------- Plot --------
    fig, ax = plt.subplots()
    ax.bar(results_df["Model"], results_df["R2"])
    ax.set_ylabel("R2 Score")
    st.pyplot(fig)
