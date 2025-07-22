# PayPredictor
It predicts the salary of the employee by their Personal Info
📌 Project Overview
Employee Salary Predictor is a machine learning web application built using Python, Scikit-learn, and Streamlit. This app predicts an employee's salary based on key features such as:

Age

Education Level

Job Title

Years of Experience

Location

The goal of this project is to assist HR professionals, job seekers, and analysts in estimating fair compensation using real-world features.

🧾 Dataset Information:
Total Records: 376 rows

Total Features: 6 columns

Columns:

1.Age (Numeric)

2.Education Level (Categorical)

3.Job Title (Categorical)

4.Years of Experience (Numeric)

5.Location (Categorical)

6.Salary (Target variable - Numeric)

The dataset was preprocessed by handling missing values, encoding categorical features, and normalizing input where needed.

🔧 Tech Stack:

. Python 3.x

. Pandas & NumPy

. Scikit-learn

. Streamlit

. Matplotlib / Seaborn (for optional visualization)

📈 Model Training:
----Model used: Random Forest Regressor

-----Features selected based on correlation and domain knowledge.

----Trained and saved using joblib.

📍 Output:-
Predicted salary displayed based on user input.

Streamlit app provides a friendly interface for real-time predictions.
