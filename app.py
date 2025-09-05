import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

# Load dataset
df = pd.read_csv("Salary Data.csv")

X = df.drop("Salary", axis=1)
y = df["Salary"]

# Preprocessing
numeric_features = ["Age", "YearsExperience"]
categorical_features = ["Job Title", "Education Level"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

# Train
model.fit(X, y)

# Streamlit interface
st.title("PayPredictor - Employee Salary Prediction")
age = st.number_input("Enter Age", min_value=18, max_value=70, value=30)
experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)
job = st.selectbox("Job Title", df["Job Title"].unique())
edu = st.selectbox("Education Level", df["Education Level"].unique())

input_df = pd.DataFrame([[age, experience, job, edu]],
                        columns=["Age", "YearsExperience", "Job Title", "Education Level"])

prediction = model.predict(input_df)[0]
st.success(f"Predicted Salary: ${prediction:,.2f}")


