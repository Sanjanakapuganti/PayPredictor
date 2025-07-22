%%writefile app.py

import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("salary_predictor.pkl")

# Streamlit UI
st.title("💼 Employee Salary Prediction App")

age = st.number_input("Age", min_value=18, max_value=70, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])
job_title = st.selectbox("Job Title", ["Software Engineer", "Data Analyst", "Senior Manager", 
                                       "Sales Associate", "Director"])  # Update with full list
experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)

# Predict button
if st.button("Predict Salary"):
    input_df = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Education Level': [education],
        'Job Title': [job_title],
        'Years of Experience': [experience]
    })

    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Salary: ${prediction:,.2f}")
#Now deploy the app using streamlit by combining app.py and salary_predictor.pkl into one folder.
#use Anaconda terminal for faster results
