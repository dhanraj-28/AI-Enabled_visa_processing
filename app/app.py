import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(page_title="Visa Processing Estimator", layout="wide")

st.title("AI Enabled Visa Processing Time Estimator")
st.markdown("Predict visa processing time using Machine Learning")

# ---------------------------------
# Load model + encoders
# ---------------------------------
model_path = "visa_model.pkl"

if os.path.exists(model_path):
    st.success("Model loaded successfully")
else:
    st.error("Model not found. Run train_model.py")
    st.stop()

model = joblib.load("visa_model.pkl")
center_encoder = joblib.load("center_encoder.pkl")
status_encoder = joblib.load("status_encoder.pkl")

# ---------------------------------
# Tabs
# ---------------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "EDA", "Prediction"])

# =================================
# TAB 1 → OVERVIEW
# =================================
with tab1:

    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("Milestone 1")
        st.write("Data cleaning and preprocessing")

    with col2:
        st.info("Milestone 2")
        st.write("EDA and Feature Engineering")

    with col3:
        st.info("Milestone 3")
        st.write("Model training and deployment")

# =================================
# TAB 2 → EDA
# =================================
with tab2:

    st.header("EDA Results")

    col1, col2 = st.columns(2)

    with col1:
        if os.path.exists("reports/processing_days_distribution.png"):
            st.image("reports/processing_days_distribution.png")
        else:
            st.warning("EDA plot not found")

    with col2:
        if os.path.exists("data/engineered_data.csv"):
            df = pd.read_csv("data/engineered_data.csv")
            st.write("Dataset Preview")
            st.dataframe(df.head())

            st.write("Statistics")
            st.dataframe(df.describe())
        else:
            st.warning("Engineered dataset not found")

# =================================
# TAB 3 → PREDICTION
# =================================
with tab3:

    st.header("Visa Processing Prediction")

    # -----------------------------
    # Input Layout
    # -----------------------------
    col1, col2 = st.columns(2)

    centers = list(center_encoder.classes_)
    statuses = list(status_encoder.classes_)

    with col1:
        selected_center = st.selectbox("Processing Center", centers)
        wage = st.number_input("Wage", min_value=0.0, value=50000.0)
        workers = st.number_input("Total Workers", min_value=1, value=1)

    with col2:
        selected_status = st.selectbox("Case Status", statuses)
        prevailing_wage = st.number_input("Prevailing Wage", min_value=1.0, value=45000.0)
        year = st.number_input("Year", min_value=2007, value=2017)

    st.markdown("---")

    # -----------------------------
    # Prediction Button
    # -----------------------------
    if st.button("Predict Processing Time"):

        # Encode categorical
        center_encoded = center_encoder.transform([selected_center])[0]
        status_encoded = status_encoder.transform([selected_status])[0]

        # Feature Engineering (same as training)
        wage_log = np.log1p(wage)
        ratio = wage / prevailing_wage
        workers_log = np.log1p(workers)
        year_trend = year - 2007

        # Temporary averages
        center_avg_time = 30
        status_avg_time = 30

        # Final input
        input_data = np.array([[
            center_encoded,
            status_encoded,
            center_avg_time,
            status_avg_time,
            wage_log,
            ratio,
            workers_log,
            year_trend
        ]])

        prediction = model.predict(input_data)

        # -----------------------------
        # Output UI
        # -----------------------------
        st.success(f"Estimated Processing Time: {int(prediction[0])} days")

        st.info("Prediction based on ML model trained on historical visa data")
