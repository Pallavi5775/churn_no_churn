import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model_path = 'best_rf_model.pkl'
model = joblib.load(model_path)

# Retrieve feature names used during training
trained_feature_names = model.feature_names_in_

# Streamlit App
st.title("Churn Prediction Dashboard")

st.markdown("### Enter the feature values for prediction")

# Define input fields for features (customize as per your dataset)
senior_citizen = st.selectbox("Senior Citizen", [0, 1])  # Binary: 0 = No, 1 = Yes
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=0)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=0.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=0.0)
contract = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber Optic", "No"])
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No"])
tech_support = st.selectbox("Tech Support", ["Yes", "No"])
device_protection = st.selectbox("Device Protection", ["Yes", "No"])

# Define mappings for categorical values
binary_mapping = {"Yes": 1, "No": 0}
contract_mapping = {"Month-to-Month": 0, "One Year": 1, "Two Year": 2}
internet_service_mapping = {"DSL": 0, "Fiber Optic": 1, "No": 2}

# Prepare initial input data
input_data = pd.DataFrame({
    "SeniorCitizen": [senior_citizen],
    "Partner": [binary_mapping[partner]],
    "Dependents": [binary_mapping[dependents]],
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "Contract": [contract_mapping[contract]],
    "InternetService": [internet_service_mapping[internet_service]],
    "PhoneService": [binary_mapping[phone_service]],
    "OnlineSecurity": [binary_mapping[online_security]],
    "TechSupport": [binary_mapping[tech_support]],
    "DeviceProtection": [binary_mapping[device_protection]]
})

# Function to preprocess input data
def preprocess_input(input_data):
    # Add "_no_transform" versions of numeric features
    input_data["tenure_no_transform"] = input_data["tenure"]
    input_data["MonthlyCharges_no_transform"] = input_data["MonthlyCharges"]
    input_data["TotalCharges_no_transform"] = input_data["TotalCharges"]

    # Add interaction terms
    input_data["DeviceProtection_div_TotalCharges^2"] = (
        input_data["DeviceProtection"] / (input_data["TotalCharges"] ** 2 + 1e-6)
    )
    input_data["TechSupport_div_Contract^2"] = (
        input_data["TechSupport"] / (input_data["Contract"] ** 2 + 1e-6)
    )

    # Add log-transformed features
    input_data["log_DeviceProtection_div_TotalCharges^2"] = np.log1p(
        input_data["DeviceProtection_div_TotalCharges^2"]
    )
    input_data["log_TechSupport_div_Contract^2"] = np.log1p(
        input_data["TechSupport_div_Contract^2"]
    )

    # Ensure all required columns are present
    for col in trained_feature_names:
        if col not in input_data.columns:
            input_data[col] = 0  # Add missing columns with default values

    # Align columns with trained feature names
    input_data = input_data[trained_feature_names]

    return input_data

# Predict button
if st.button("Predict"):
    try:
        # Preprocess the input data
        input_data = preprocess_input(input_data)

        # Make predictions
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[:, 1]

        # Display results
        st.write("### Prediction Results")
        st.write(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
        st.write(f"Probability of Churn: {probability[0]:.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
