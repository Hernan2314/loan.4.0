import pickle  # Importing the pickle library
import joblib  # For loading the scaler
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io  # For in-memory CSV download

# Load the trained model
pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in)

# Load the scaler
scaler = joblib.load('scaler.pkl')

@st.cache_data()
def prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History):   
    # Pre-process user input
    Gender = 0 if Gender == "Male" else 1
    Married = 0 if Married == "Unmarried" else 1
    Credit_History = 0 if Credit_History == "Unclear Debts" else 1
    LoanAmount = LoanAmount / 1000  # Scale loan amount if required

    # Create the feature array
    features = np.array([Gender, Married, ApplicantIncome, LoanAmount, Credit_History]).reshape(1, -1)

    # Scale the features using the loaded scaler
    scaled_features = scaler.transform(features)

    # Predict with scaled features
    prediction = classifier.predict(scaled_features)
    return 'Approved' if prediction == 1 else 'Rejected'

# Streamlit Interface
def main():
    st.set_page_config(page_title="ProFund Insight - Loan Approval", page_icon="💼", layout="centered")

    # Branding and Title
    st.markdown("""
        <style>
        .title { font-size: 2.4em; font-weight: bold; color: #2e3a45; }
        .subtitle { font-size: 1.2em; color: #6c757d; }
        </style>
        """, unsafe_allow_html=True)
    st.markdown('<p class="title">💼 ProFund Insight - Loan Approval System</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">A Trusted Solution for Modern Financial Decision Making</p>', unsafe_allow_html=True)

    # Input Form
    st.markdown("### Application Details")
    Gender = st.radio("Select your Gender:", ("Male", "Female"), help="Choose the gender of the applicant.")
    Married = st.radio("Marital Status:", ("Unmarried", "Married"), help="Choose the marital status of the applicant.")
    ApplicantIncome = st.slider("Applicant's Monthly Income (in USD)", min_value=0, max_value=20000, step=500, value=5000, help="Enter the applicant's monthly income.")
    LoanAmount = st.slider("Loan Amount Requested (in USD)", min_value=0, max_value=500000, step=1000, value=150000, help="Enter the loan amount the applicant is requesting.")
    Credit_History = st.selectbox("Credit History Status:", ("Unclear Debts", "No Unclear Debts"), help="Select the applicant's credit history status.")

    # Prediction Button
    if st.button("Predict My Loan Status"):
        result = prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History)
        
        # Display Prediction Outcome
        if result == "Approved":
            st.success(f'✅ Your loan application status: **Approved**')
        else:
            st.error(f'❌ Your loan application status: **Rejected**')

        # Executive Summary Section
        st.write("---")
        st.subheader("Executive Summary")
        st.write(f"""
            **Applicant Details**
            - **Gender**: {Gender}
            - **Marital Status**: {Married}
            - **Monthly Income**: ${ApplicantIncome}
            - **Loan Amount Requested**: ${LoanAmount}
            - **Credit History**: {Credit_History}

            **Decision**: The loan application was **{result}** based on the applicant's profile and historical approval criteria.
        """)

        # Additional visualization and report generation sections follow here...

if __name__ == '__main__':
    main()
