import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("wine_quality_model.pkl", "rb"))

st.set_page_config(page_title="Wine Quality Prediction", layout="centered")

st.title("🍷 Wine Quality Prediction App")
st.write("Enter the chemical properties of wine to predict its quality")

# Input fields
fixed_acidity = st.number_input("Fixed Acidity", value=7.9)
volatile_acidity = st.number_input("Volatile Acidity", value=0.35)
citric_acid = st.number_input("Citric Acid", value=0.46)
residual_sugar = st.number_input("Residual Sugar", value=3.6)
chlorides = st.number_input("Chlorides", value=0.078)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=15)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=37)
density = st.number_input("Density", value=0.9973, format="%.4f")
pH = st.number_input("pH", value=3.35)
sulphates = st.number_input("Sulphates", value=0.86)
alcohol = st.number_input("Alcohol", value=12.8)

# Predict button
if st.button("Predict Wine Quality"):
    sample_data = np.array([[fixed_acidity,
                              volatile_acidity,
                              citric_acid,
                              residual_sugar,
                              chlorides,
                              free_sulfur_dioxide,
                              total_sulfur_dioxide,
                              density,
                              pH,
                              sulphates,
                              alcohol]])

    prediction = model.predict(sample_data)[0]

    st.success(f"🍷 Predicted Wine Quality Score: **{prediction}**")

    if prediction >= 7:
        st.write("✅ This is a **Good Quality Wine**")
    elif prediction >= 5:
        st.write("⚠️ This is an **Average Quality Wine**")
    else:
        st.write("❌ This is a **Low Quality Wine**")
