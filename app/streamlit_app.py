# app/streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("Titanic Survival Predictor")
st.write("Interactive demo for the Titanic survival classifier (traditional ML).")

# Load artifacts
preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("model.pkl")

# Sidebar inputs
st.sidebar.header("Passenger attributes")
pclass = st.sidebar.selectbox("Passenger class (pclass)", [1,2,3], index=2)
sex = st.sidebar.selectbox("Sex", ["male","female"], index=0)
age = st.sidebar.slider("Age", 0.0, 100.0, 30.0)
sibsp = st.sidebar.slider("Siblings/Spouses aboard", 0, 8, 0)
parch = st.sidebar.slider("Parents/Children aboard", 0, 6, 0)
fare = st.sidebar.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0)
embarked = st.sidebar.selectbox("Port of Embarkation", ["C","Q","S"], index=2)

if st.sidebar.button("Predict survival"):
    # Build input dataframe (same column names as training)
    input_df = pd.DataFrame([{
        'pclass': pclass,
        'sex': sex,
        'age': age,
        'sibsp': sibsp,
        'parch': parch,
        'fare': fare,
        'embarked': embarked
    }])
    # Preprocess & predict
    X_proc = preprocessor.transform(input_df)
    proba = model.predict_proba(X_proc)[:,1][0]
    pred = int(proba >= 0.5)

    st.metric("Survival probability", f"{proba*100:.1f}%")
    st.success("Prediction: Survived" if pred==1 else "Prediction: Did NOT survive")
    st.write("Model probabilities and prediction threshold (0.5).")
else:
    st.info("Set passenger attributes in the sidebar and click Predict.")
