# STREAMLIT DEPLOYMENT FOR CHURN CLASSIFICATION

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# === Load saved model and scaler ===
with open("final_churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler_churn.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# === Streamlit App ===
st.title("Customer Churn Prediction")

# Input fields
gender = st.selectbox("Gender", ['Female', 'Male'])
partner = st.selectbox("Has Partner?", ['Yes', 'No'])
dependents = st.selectbox("Has Dependents?", ['Yes', 'No'])
phone_service = st.selectbox("Has Phone Service?", ['Yes', 'No'])
paperless_billing = st.selectbox("Paperless Billing?", ['Yes', 'No'])

tenure = st.number_input("Tenure (months)", min_value=0)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
payment_method = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

# Build input dictionary
input_dict = {
    'gender': 1 if gender == 'Male' else 0,
    'Partner': 1 if partner == 'Yes' else 0,
    'Dependents': 1 if dependents == 'Yes' else 0,
    'PhoneService': 1 if phone_service == 'Yes' else 0,
    'PaperlessBilling': 1 if paperless_billing == 'Yes' else 0,
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'InternetService_DSL': 1 if internet_service == 'DSL' else 0,
    'InternetService_Fiber optic': 1 if internet_service == 'Fiber optic' else 0,
    'InternetService_No': 1 if internet_service == 'No' else 0,
    'Contract_Month-to-month': 1 if contract == 'Month-to-month' else 0,
    'Contract_One year': 1 if contract == 'One year' else 0,
    'Contract_Two year': 1 if contract == 'Two year' else 0,
    'PaymentMethod_Electronic check': 1 if payment_method == 'Electronic check' else 0,
    'PaymentMethod_Mailed check': 1 if payment_method == 'Mailed check' else 0,
    'PaymentMethod_Bank transfer (automatic)': 1 if payment_method == 'Bank transfer (automatic)' else 0,
    'PaymentMethod_Credit card (automatic)': 1 if payment_method == 'Credit card (automatic)' else 0,
}

# Set default 0 for all other features
input_df = pd.DataFrame(columns=feature_columns)
input_df.loc[0] = 0  # fill with 0s
for key in input_dict:
    if key in input_df.columns:
        input_df.at[0, key] = input_dict[key]

# Scale numeric columns
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# Predict
if st.button("Predict Churn"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    if pred == 1:
        st.error(f"❌ This customer is likely to churn. Probability: {prob:.2f}")
    else:
        st.success(f"✅ This customer is likely to stay. Probability: {prob:.2f}")
