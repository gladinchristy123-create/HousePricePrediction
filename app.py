import math
import json
import streamlit as st
import pandas as pd
import joblib
import sqlite3
import os
from datetime import datetime

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

with open("models/model_metadata.json", "r", encoding="utf-8") as file:
    MODEL_VERSION = json.load(file)["model_version"]

# Database setup
def initialize_database():
    conn = sqlite3.connect("data/predictions.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            model_version TEXT,
            model_name TEXT,
            median_income REAL,
            house_age REAL,
            average_rooms REAL,
            average_bedrooms REAL,
            population REAL,
            average_occupancy REAL,
            latitude REAL,
            longitude REAL,
            predicted_price REAL
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(values, model_name, prediction):
    conn = sqlite3.connect("data/predictions.db")
    conn.execute("""
        INSERT INTO predictions (
            timestamp, model_version, model_name,
            median_income, house_age, average_rooms,
            average_bedrooms, population, average_occupancy,
            latitude, longitude, predicted_price
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(timespec="seconds"),
        MODEL_VERSION,
        model_name,
        *values,
        float(prediction)
    ))
    conn.commit()
    conn.close()


initialize_database()

# Load model comparison
st.title("🏠 California House Price Prediction")
st.write("Machine Learning Based House Price Estimation Application")

st.caption(f"Deployed model pipeline version: {MODEL_VERSION}")

try:
    comparison = pd.read_csv("results/model_comparison.csv")
except FileNotFoundError:
    st.error("Model comparison file not found. Run train.py first.")
    st.stop()

# Sidebar
st.sidebar.header("House Details")

median_income = st.sidebar.number_input(
    "Median Income (in $10,000s)", 0.5, 20.0, 5.0
)

house_age = st.sidebar.number_input(
    "House Age (years)", 1.0, 52.0, 20.0
)

average_rooms = st.sidebar.number_input(
    "Average Rooms", 1.0, 20.0, 5.0
)

average_bedrooms = st.sidebar.number_input(
    "Average Bedrooms", 0.5, 10.0, 1.0
)

population = st.sidebar.number_input(
    "Population", 1.0, 50000.0, 1000.0
)

average_occupancy = st.sidebar.number_input(
    "Average Occupancy", 1.0, 10.0, 3.0
)

latitude = st.sidebar.number_input(
    "Latitude", 32.0, 42.0, 34.0
)

longitude = st.sidebar.number_input(
    "Longitude", -125.0, -114.0, -118.0
)

model_choice = st.sidebar.selectbox(
    "Choose Prediction Model",
    ["Linear Regression", "Ridge Regression", "Random Forest"]
)


if st.sidebar.button("Predict House Price"):
    model_files = {
        "Linear Regression": "models/linear_regression.joblib",
        "Ridge Regression": "models/ridge_regression.joblib",
        "Random Forest": "models/random_forest.joblib"
    }

    model = joblib.load(model_files[model_choice])

    values = [
        median_income,
        house_age,
        average_rooms,
        average_bedrooms,
        population,
        average_occupancy,
        latitude,
        longitude
    ]

    input_data = pd.DataFrame(
        [values],
        columns=[
            "MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"
        ]
    )

    prediction = model.predict(input_data)[0]

    # Model monitoring check
    if not math.isfinite(prediction) or prediction < 0:
        st.error("Monitoring Alert: Invalid model prediction detected.")
        st.stop()

    save_prediction(values, model_choice, prediction)

    st.success("Prediction completed successfully!")

    st.metric(
        "Estimated Median House Value",
        f"${prediction:,.2f}"
    )

    st.info(f"Model: {model_choice} | Version: {MODEL_VERSION}")


# Model comparison
st.header("📊 Model Performance Comparison")

st.dataframe(comparison, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("RMSE Comparison")
    st.bar_chart(comparison.set_index("Model")["RMSE"])

with col2:
    st.subheader("R² Score Comparison")
    st.bar_chart(comparison.set_index("Model")["R2"])

# Prediction history
st.header("📁 Saved Prediction History")

conn = sqlite3.connect("data/predictions.db")
history = pd.read_sql_query(
    "SELECT * FROM predictions ORDER BY id DESC",
    conn
)
conn.close()

if not history.empty:
    st.dataframe(history, use_container_width=True)
else:
    st.write("No predictions saved yet.")

st.caption(
    "Note: This application estimates median house values for "
    "California census block groups, not exact individual property prices."
)
