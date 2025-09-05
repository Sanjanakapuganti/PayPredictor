import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load data
df = pd.read_csv("Salary Data.csv")

# Drop rows with missing values
df.dropna(inplace=True)

# Features and target
X = df.drop("Salary", axis=1)
y = df["Salary"]

# Define preprocessing
numerical_features = ["Age", "Years of Experience"]
categorical_features = ["Gender", "Education Level", "Job Title"]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Create pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'salary_predictor.pkl')

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
