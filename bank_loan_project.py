import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBClassifier

# 1. Ingestion & Preprocessing
df = pd.read_csv("loan_dataset_20000.csv").drop_duplicates()
pd.set_option("display.max_row",None)
pd.set_option("display.max_columns",None)
# Domain Feature Engineering
df1 = df.columns
for i in df1:
    df = df.rename(columns={i:i.capitalize()})
print(df.head(4))
df["loan_year_term"] = df["Loan_term"] / 12
df["Loan_To_Income"] = df["Loan_amount"] / df["Annual_income"]
df["Total_credit_ratio"] = df["Total_credit_limit"] / df["Annual_income"]
df["Risky_pay_loan"] = (
    df["Delinquency_history"]
    + df["Public_records"]
    + df["Num_of_delinquencies"]
)

# Custom Business Risk Flags
df["Is_High_Risk"] = 0
df["Is_low_Risk"] = 0

cond_high = (df["Employment_status"] == "Unemployment") & (
    df["Credit_score"] < 790
)
cond_low = (
    (df["Employment_status"] != "Unemployment")
    & (df["Credit_score"] > 790)
    & (df["Annual_income"] > 35000)
    & (df["Loan_amount"] < 25000)
)

df.loc[cond_high, "Is_High_Risk"] = 1
df.loc[cond_low, "Is_low_Risk"] = 1

df["Credit_utilization_rate"] = df["Current_balance"] / (
    df["Total_credit_limit"] + 1
)
df["EMI-to-Monthli_ratio"] = df["Installment"] / (df["Monthly_income"] + 1)
df["Annual_income_ratio_credit_score"] = (
    df["Annual_income"] / df["Credit_score"]
)
df["Long_year_real"] = (df["Loan_amount"] / df["Installment"]) / 12
df["Monthly_Debt_Expense"] = (
    df["Debt_to_income_ratio"] * df["Monthly_income"]
)
df["Annual_Debt_Expense"] = df["Monthly_Debt_Expense"] * 12
df["Monthly_Debt_Expence_ratio"] = (
    df["Monthly_income"] / df["Monthly_Debt_Expense"] + 1
)
df["Annual_Debt_Expence_ratio"] = (
    df["Annual_income"] / df["Monthly_Debt_Expense"] + 1
)

# 2. Features & Target
X = df.drop("Loan_paid_back", axis=1)
y = df["Loan_paid_back"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Categorical Encodings Configuration
grade_order = [
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "E1",
    "E2",
    "E3",
    "E4",
    "E5",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
]
edu_order = ["Other", "High School", "Bachelor's", "Master's", "PhD"]

ordinal_categories = [grade_order, edu_order]
ordinal_features = ["Grade_subgrade", "Education_level"]
nominal_columns = [
    "Gender",
    "Marital_status",
    "Employment_status",
    "Loan_purpose",
]
numerical_columns = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

# 3. Pipelines Setup
num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])

ohe_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
])

ordinal_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    (
        "ordinal",
        OrdinalEncoder(
            categories=ordinal_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        ),
    ),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipe, numerical_columns),
        ("ordinal", ordinal_pipe, ordinal_features),
        ("ohe", ohe_pipe, nominal_columns),
    ]
)

# 4. XGBoost Classifier Engine
model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.02,
    max_depth=7,
    subsample=0.8,
    reg_alpha=1.0,
    reg_lambda=1.0,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    scale_pos_weight=3.9975,
    n_jobs=-1,
)

master_pipeline = Pipeline([
    ("Preprocessor", preprocessor),
    ("model", model),
])

# 5. Model Training & Business Metric Evaluation
master_pipeline.fit(X_train, y_train)

# Probability Threshold Tuning
y_probs = master_pipeline.predict_proba(X_test)[:, 1]
custom_threshold = 0.65
y_pred_custom = (y_probs >= custom_threshold).astype(int)
print("-------- Train and Test score-------")
print("Train Score:", master_pipeline.score(X_train,y_train)*100)
print("Test Score:", master_pipeline.score(X_test,y_test)*100)
print(f"--- XGBoost Model Evaluation (Threshold = {custom_threshold}) ---")
print(
    "Precision:",
    round(precision_score(y_test, y_pred_custom) * 100, 2),
    "%",
)
print("Recall:", round(recall_score(y_test, y_pred_custom) * 100, 2), "%")
print("F1-Score:", round(f1_score(y_test, y_pred_custom) * 100, 2), "%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_custom))

# 6. Feature Importance Extraction
feature_names = master_pipeline.named_steps[
    "Preprocessor"
].get_feature_names_out()
importances = master_pipeline.named_steps["model"].feature_importances_

feat_imp_df = pd.DataFrame(
    {"Feature": feature_names, "Importance": importances}
).sort_values(by="Importance", ascending=False)

print("\n--- Top Features Driving the Business Model ---")
print(feat_imp_df.head(10))
print("profit clients  of net bad 10 ,15 crore")
import joblib

joblib.dump(master_pipeline, "loan_npa_xgboost_pipeline.pkl")
