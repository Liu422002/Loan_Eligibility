import streamlit as st
import pandas as pd
from src.save_model import load_model

st.title("Loan Eligibility Prediction App")
st.write("Enter the applicant details below to predict loan approval.")

try:
    model = load_model("models/tuned_random_forest_model.pkl")
    scaler = load_model("models/scaler.pkl")

    applicant_income = st.number_input("Applicant Income", min_value=0.0, value=5000.0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0)
    loan_amount = st.number_input("Loan Amount", min_value=0.0, value=150.0)
    loan_amount_term = st.number_input("Loan Amount Term", min_value=0.0, value=360.0)
    credit_history = st.selectbox("Credit History", [0.0, 1.0])

    gender_male = st.selectbox("Gender Male", [0, 1])
    married_yes = st.selectbox("Married Yes", [0, 1])

    dependents_1 = st.selectbox("Dependents 1", [0, 1])
    dependents_2 = st.selectbox("Dependents 2", [0, 1])
    dependents_3_plus = st.selectbox("Dependents 3+", [0, 1])

    education_not_graduate = st.selectbox("Education Not Graduate", [0, 1])
    self_employed_yes = st.selectbox("Self Employed Yes", [0, 1])

    property_area_semiurban = st.selectbox("Property Area Semiurban", [0, 1])
    property_area_urban = st.selectbox("Property Area Urban", [0, 1])

    if st.button("Predict"):
        input_data = pd.DataFrame([{
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_amount_term,
            "Credit_History": credit_history,
            "Gender_Male": gender_male,
            "Married_Yes": married_yes,
            "Dependents_1": dependents_1,
            "Dependents_2": dependents_2,
            "Dependents_3+": dependents_3_plus,
            "Education_Not Graduate": education_not_graduate,
            "Self_Employed_Yes": self_employed_yes,
            "Property_Area_Semiurban": property_area_semiurban,
            "Property_Area_Urban": property_area_urban
        }])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
            st.success("Loan Approved")
        else:
            st.error("Loan Not Approved")

except Exception as e:
    st.error(f"Error loading model or making prediction: {e}")