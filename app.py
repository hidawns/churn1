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

# === Input: Binary Features ===
gender = st.selectbox("Gender", ['Female', 'Male'])
senior_citizen = st.selectbox("Is Senior Citizen?", ['Yes', 'No'])
partner = st.selectbox("Has Partner?", ['Yes', 'No'])
dependents = st.selectbox("Has Dependents?", ['Yes', 'No'])
phone_service = st.selectbox("Has Phone Service?", ['Yes', 'No'])
paperless_billing = st.selectbox("Uses Paperless Billing?", ['Yes', 'No'])

# === Input: Multi-category ===
multiple_lines = st.selectbox("Multiple Lines", ['No', 'Yes', 'No phone service'])
internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
online_security = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
online_backup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
device_protection = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
tech_support = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
streaming_tv = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
streaming_movies = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
payment_method = st.selectbox("Payment Method", [
    'Electronic check', 'Mailed check', 
    'Bank transfer (automatic)', 'Credit card (automatic)'
])

# === Input: Numeric Features ===
tenure = st.number_input("Tenure (months)", min_value=0)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

# === Build input dict ===
input_dict = {
    #binary features
    'gender': 1 if gender == 'Male' else 0,
    'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
    'Partner': 1 if partner == 'Yes' else 0,
    'Dependents': 1 if dependents == 'Yes' else 0,
    'PhoneService': 1 if phone_service == 'Yes' else 0,
    'PaperlessBilling': 1 if paperless_billing == 'Yes' else 0,

    # One-hot encoded features (initialize below)
    f"MultipleLines_{multiple_lines}": 1,
    f"InternetService_{internet_service}": 1,
    f"OnlineSecurity_{online_security}": 1,
    f"OnlineBackup_{online_backup}": 1,
    f"DeviceProtection_{device_protection}": 1,
    f"TechSupport_{tech_support}": 1,
    f"StreamingTV_{streaming_tv}": 1,
    f"StreamingMovies_{streaming_movies}": 1,
    f"Contract_{contract}": 1,
    f"PaymentMethod_{payment_method}": 1,

    # numeric features
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
}

# === Build input dataframe ===
input_df = pd.DataFrame(columns=feature_columns)
input_df.loc[0] = 0  # Default all 0s

# Fill in provided input
for key in input_dict:
    if key in input_df.columns:
        input_df.at[0, key] = input_dict[key]

# Scale numeric features
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# === Predict ===
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"❌ Customer likely to churn. Probability to churn: {probability:.2f}")
    else:
        st.success(f"✅ Customer likely to stay. Probability to churn: {probability:.2f}")
