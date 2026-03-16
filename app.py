import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("loan_model.pkl","rb"))

st.set_page_config(page_title="Loan Approval Predictor", page_icon="💰")

st.title("💰 Credit Wise Loan Approval System")
st.write("Fill the details below to check whether your loan is likely to be approved.")

st.markdown("---")

# PERSONAL DETAILS
st.subheader("👤 Personal Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=70, step=1)
    gender = st.selectbox("Gender", ["Female","Male"])
    marital_status = st.selectbox("Marital Status", ["Married","Single"])
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=10)

with col2:
    education = st.selectbox("Education Level", ["Graduate","Not Graduate"])
    employment = st.selectbox(
        "Employment Status",
        ["Salaried","Self-employed","Unemployed"]
    )
    employer_category = st.selectbox(
        "Employer Category",
        ["Private","Government","MNC","Unemployed"]
    )

st.markdown("---")

# FINANCIAL DETAILS
st.subheader("💵 Financial Information")

col3, col4 = st.columns(2)

with col3:
    applicant_income = st.number_input(
        "Applicant Monthly Income (₹)", min_value=0
    )

    coapplicant_income = st.number_input(
        "Co-applicant Monthly Income (₹)", min_value=0
    )

    savings = st.number_input(
        "Total Savings (₹)",
        help="Total money saved in bank accounts"
    )

    existing_loans = st.number_input(
        "Number of Existing Loans",
        min_value=0
    )

with col4:
    loan_amount = st.number_input(
        "Loan Amount Requested (₹)"
    )

    loan_term = st.slider(
        "Loan Term (Years)",
        1, 30
    )

    collateral_value = st.number_input(
        "Collateral Value (₹)",
        help="Value of property or asset used as security"
    )

st.markdown("---")

# PROPERTY + PURPOSE
st.subheader("🏠 Loan Details")

loan_purpose = st.selectbox(
    "Loan Purpose",
    ["Home","Car","Education","Personal"]
)

property_area = st.selectbox(
    "Property Area",
    ["Urban","Semiurban","Rural"]
)

credit_score = st.number_input(
    "Credit Score (300 - 900)",
    min_value=300,
    max_value=900
)

st.markdown("---")

# Feature Engineering
credit_score_sq = credit_score ** 2
dti_ratio_sq = ((loan_amount) / (applicant_income + 1)) ** 2

# Encoding
gender_male = 1 if gender == "Male" else 0
marital_single = 1 if marital_status == "Single" else 0
education_level = 1 if education == "Graduate" else 0

# Loan purpose encoding
loan_car = 1 if loan_purpose == "Car" else 0
loan_edu = 1 if loan_purpose == "Education" else 0
loan_home = 1 if loan_purpose == "Home" else 0
loan_personal = 1 if loan_purpose == "Personal" else 0

# Property area encoding
prop_semiurban = 1 if property_area == "Semiurban" else 0
prop_urban = 1 if property_area == "Urban" else 0

# Employment encoding
emp_sal = 1 if employment == "Salaried" else 0
emp_self = 1 if employment == "Self-employed" else 0
emp_unemp = 1 if employment == "Unemployed" else 0

# Employer category
emp_gov = 1 if employer_category == "Government" else 0
emp_mnc = 1 if employer_category == "MNC" else 0
emp_priv = 1 if employer_category == "Private" else 0
emp_un = 1 if employer_category == "Unemployed" else 0

# Final feature array
features = np.array([[applicant_income, coapplicant_income, age, dependents,
existing_loans, savings, collateral_value, loan_amount,
loan_term, education_level, marital_single,
loan_car, loan_edu, loan_home, loan_personal,
prop_semiurban, prop_urban,
emp_sal, emp_self, emp_unemp,
gender_male,
emp_gov, emp_mnc, emp_priv, emp_un,
credit_score_sq, dti_ratio_sq]])

# Prediction button
if st.button("🔍 Predict Loan Approval"):

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.success("✅ Loan is likely to be Approved")
    else:
        st.error("❌ Loan may NOT be Approved")