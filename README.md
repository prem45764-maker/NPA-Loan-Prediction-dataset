# NPA-Loan-Prediction-dataset
Loan NPA Prediction Project – README
🔹 Overview
This project builds a machine learning pipeline to predict whether a loan will be paid back or become a Non‑Performing Asset (NPA). It uses XGBoost Classifier with advanced feature engineering and categorical encodings to achieve high accuracy in risk prediction.

🔹 Features
Data Preprocessing:

Duplicate removal, column renaming, missing value imputation.

Feature engineering (Loan‑to‑Income ratio, Credit Utilization Rate, Debt Expense ratios).

Risk Flags:

Custom rules to classify borrowers as High Risk or Low Risk.

Model Training:

XGBoost Classifier with tuned hyperparameters.

Threshold tuning for better precision/recall balance.

Evaluation Metrics:

Precision, Recall, F1‑Score, Confusion Matrix.

Feature importance extraction for business insights.

🔹 Tech Stack
Python

Libraries: NumPy, Pandas, Scikit‑Learn, LightGBM, XGBoost, Joblib

🔹 Results
Achieved strong performance with custom threshold (0.65).

Precision, Recall, and F1‑Score balanced for business risk detection.

Top features driving predictions include Credit Score, Loan Amount, Debt Ratios.

Business insight: Potential profit recovery estimated at ₹10–15 crore by identifying risky clients early.

🔹 How to Run
Clone repository and place loan_dataset_20000.csv in project folder.

Install dependencies:

bash
pip install -r requirements.txt
Run training script:

bash
python loan_npa_pipeline.py
Saved model: loan_npa_xgboost_pipeline.pkl for deployment.

🔹 Future Scope
Deploy model with Streamlit dashboard for interactive loan risk analysis.

Integrate with banking systems for real‑time credit decisioning.
