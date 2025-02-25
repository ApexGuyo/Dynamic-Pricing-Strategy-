import streamlit as st
import numpy as np
import joblib


def dynamic_pricing(base_price, demand, supply, competitor_price, model):
    # Use trained ML model for prediction
    features = np.array([[base_price, demand, supply, competitor_price]])
    new_price = model.predict(features)[0]
    return round(new_price, 2)


# Load pre-trained model
model = joblib.load("pricing_model.pkl")

# Streamlit UI
st.title("AI-Powered Dynamic Pricing Strategy")
st.sidebar.header("Input Parameters")

base_price = st.sidebar.number_input(
    "Base Price (Ksh)", min_value=100, max_value=10000, value=1000)
demand = st.sidebar.number_input(
    "Demand Level", min_value=1, max_value=1000, value=500)
supply = st.sidebar.number_input(
    "Supply Level", min_value=1, max_value=1000, value=500)
competitor_price = st.sidebar.number_input(
    "Competitor Price (Ksh)", min_value=100, max_value=10000, value=1000)

# Calculate new price using AI model
new_price = dynamic_pricing(
    base_price, demand, supply, competitor_price, model)

# Display results
st.subheader("Optimized Selling Price")
st.metric(label="New Price (Ksh)", value=new_price)
