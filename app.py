import streamlit as st
import pandas as pd
from src.save_model import load_model

st.title("Loan Eligibility Prediction App")
st.write("Enter the applicant details below to predict loan approval.")

try:
    model = load_model("models/tuned_random_forest_model.pkl")
    scaler = load_model("models/scaler.pkl")

    processed_df = pd.read_csv("data/Processed_Credit_Dataset.csv")
    feature_columns = processed_df.drop("Loan_Approved", axis=1).columns

    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

    applicant_income = st.number_input("Applicant Income", min_value=0.0, value=5000.0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0)
    loan_amount = st.number_input("Loan Amount", min_value=0.0, value=150.0)
    loan_amount_term = st.number_input("Loan Amount Term", min_value=0.0, value=360.0)
    credit_history = st.selectbox("Credit History", [1.0, 0.0])

    if st.button("Predict"):
        input_data = pd.DataFrame([{
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "Property_Area": property_area,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_amount_term,
            "Credit_History": credit_history
        }])

        input_encoded = pd.get_dummies(
            input_data,
            columns=[
                "Gender",
                "Married",
                "Dependents",
                "Education",
                "Self_Employed",
                "Property_Area"
            ],
            dtype=int
        )

        input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

        input_scaled = scaler.transform(input_encoded)
        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
            st.success("Loan Approved")
        else:
            st.error("Loan Not Approved")

except Exception as e:
    st.error(f"Error loading model or making prediction: {e}")