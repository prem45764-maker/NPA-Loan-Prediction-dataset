import joblib
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Loan Risk & NPA Prediction System",
    page_icon="🏦",
    layout="wide",
)

# Load Trained XGBoost Pipeline Model (.pkl)
@st.cache_resource
def load_model():
    return joblib.load("loan_npa_xgboost_pipeline.pkl")

model = load_model()
print(model.feature_names_in_)
st.title("🏦 Loan Risk & NPA Prediction System")
st.write(
    "Enterprise Risk Engine: Analyze individual loan applications or batch "
    "process portfolios using your trained XGBoost Pipeline."
)

tab1, tab2 = st.tabs(["👤 Single Customer Audit", "📁 Bulk CSV Batch Audit"])


def engineer_features(df):
    df = df.copy()

    # 1. Capitalize initial standard columns just like training dataset ingestion
    df.columns = [str(i).capitalize() for i in df.columns]

    # 2. Exact engineered features matching model.feature_names_in_ casing
    df["loan_year_term"] = df["Loan_term"] / 12
    df["Loan_To_Income"] = df["Loan_amount"] / (df["Annual_income"] + 1)
    df["Total_credit_ratio"] = df["Total_credit_limit"] / (df["Annual_income"] + 1)

    if "Delinquency_history" in df.columns and df["Delinquency_history"].dtype == "object":
        delinq_map = {"None": 0, "Minor": 1, "Severe": 2, "none": 0}
        df["Delinquency_history"] = df["Delinquency_history"].map(delinq_map).fillna(0)

    df["Risky_pay_loan"] = (
        df["Delinquency_history"]
        + df["Public_records"]
        + df["Num_of_delinquencies"]
    )

    # Custom Business Risk Flags (Exact casing from model_feature_names)
    df["Is_High_Risk"] = 0
    df["Is_Low_Risk"] = 0

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

    df["Monthly_income"] = df["Annual_income"] / 12

    df["Credit_utilization_rate"] = df["Current_balance"] / (
        df["Total_credit_limit"] + 1
    )
    
    # Exact spelling and casing mapping from feature_names_in_ list
    df["EMI-to-Monthli_ratio"] = df["Installment"] / (df["Monthly_income"] + 1)
    df["Annual_income_ratio_credit_score"] = (
        df["Annual_income"] / (df["Credit_score"] + 1)
    )
    df["Long_year_real"] = (df["Loan_amount"] / (df["Installment"] + 1)) / 12
    
    df["Monthly_Debt_Expense"] = (
        df["Debt_to_income_ratio"] * df["Monthly_income"]
    )
    df["Annual_Debt_Expense"] = df["Monthly_Debt_Expense"] * 12

    df["Monthly_Debt_Expence_ratio"] = (
        df["Monthly_income"] / (df["Monthly_Debt_Expense"] + 1e-5)
    )
    df["Annual_Debt_Expence_ratio"] = (
        df["Annual_income"] / (df["Monthly_Debt_Expense"] + 1e-5)
    )

    return df


# ==================== TAB 1: SINGLE CUSTOMER AUDIT ====================
with tab1:
    st.sidebar.header("Customer & Loan Details Input")

    st.sidebar.subheader("📊 Financial & Numerical Inputs")
    age = st.sidebar.number_input(
        "Age", min_value=18, max_value=100, value=30, step=1
    )
    annual_income = st.sidebar.number_input(
        "Annual Income ($)",
        min_value=1000.0,
        max_value=500000.0,
        value=50000.0,
        step=500.0,
    )
    loan_amount = st.sidebar.number_input(
        "Loan Amount ($)",
        min_value=500.0,
        max_value=100000.0,
        value=15000.0,
        step=100.0,
    )
    loan_term = st.sidebar.selectbox("Loan Term (Months)", [36, 60])
    credit_score = st.sidebar.number_input(
        "Credit Score", min_value=300, max_value=850, value=700, step=1
    )
    installment = st.sidebar.number_input(
        "Installment", min_value=10.0, max_value=5000.0, value=400.0, step=10.0
    )
    total_credit_limit = st.sidebar.number_input(
        "Total Credit Limit",
        min_value=0.0,
        max_value=1000000.0,
        value=50000.0,
        step=1000.0,
    )
    current_balance = st.sidebar.number_input(
        "Current Balance",
        min_value=0.0,
        max_value=500000.0,
        value=10000.0,
        step=100.0,
    )
    dti_ratio = st.sidebar.number_input(
        "Debt-to-Income Ratio (DTI)",
        min_value=0.0,
        max_value=100.0,
        value=15.0,
        step=0.1,
    )
    interest_rate = st.sidebar.number_input(
        "Interest Rate (%)", min_value=0.0, max_value=40.0, value=10.5, step=0.1
    )
    public_records = st.sidebar.number_input(
        "Public Records", min_value=0, max_value=10, value=0, step=1
    )
    num_delinq = st.sidebar.number_input(
        "Num of Delinquencies", min_value=0, max_value=20, value=0, step=1
    )
    num_open_accounts = st.sidebar.number_input(
        "Num of Open Accounts", min_value=0, max_value=50, value=5, step=1
    )

    st.sidebar.markdown("---")

    st.sidebar.subheader("📋 Profile & History Categories")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    marital_status = st.sidebar.selectbox(
        "Marital Status", ["Married", "Single", "Divorced", "Widowed"]
    )
    education_level = st.sidebar.selectbox(
        "Education Level", ["Master's", "Bachelor's", "High School", "Other", "PhD"]
    )
    employment_status = st.sidebar.selectbox(
        "Employment Status",
        ["Employed", "Unemployed", "Self-employed", "Student", "Retired"],
    )
    loan_purpose = st.sidebar.selectbox(
        "Loan Purpose",
        [
            "Car",
            "Debt consolidation",
            "Business",
            "Other",
            "Home",
            "Medical",
            "Education",
            "Vacation",
        ],
    )

    grade_subgrade_list = [
        "A1", "A2", "A3", "A4", "A5",
        "B1", "B2", "B3", "B4", "B5",
        "C1", "C2", "C3", "C4", "C5",
        "D1", "D2", "D3", "D4", "D5",
        "E1", "E2", "E3", "E4", "E5",
        "F1", "F2", "F3", "F4", "F5",
    ]
    grade_subgrade = st.sidebar.selectbox(
        "Grade Subgrade", grade_subgrade_list
    )

    delinquency_history = st.sidebar.number_input(
        "Delinquency History Count", min_value=0, max_value=20, value=0
    )

    if st.button("🚀 Predict Single Loan Status", type="primary"):
        input_data = pd.DataFrame({
            "Age": [age],
            "Annual_income": [annual_income],
            "Loan_amount": [loan_amount],
            "Loan_term": [loan_term],
            "Credit_score": [credit_score],
            "Installment": [installment],
            "Total_credit_limit": [total_credit_limit],
            "Current_balance": [current_balance],
            "Debt_to_income_ratio": [dti_ratio],
            "Interest_rate": [interest_rate],  # Fixed matching training column name
            "Gender": [gender],
            "Marital_status": [marital_status],
            "Education_level": [education_level],
            "Employment_status": [employment_status],
            "Loan_purpose": [loan_purpose],
            "Grade_subgrade": [grade_subgrade],
            "Delinquency_history": [delinquency_history],
            "Public_records": [public_records],
            "Num_of_delinquencies": [num_delinq],
            "Num_of_open_accounts": [num_open_accounts],
        })

        # Apply exact feature engineering pipeline correctly
        processed_input = engineer_features(input_data)

        # Predict using pipeline model
        expected = set(model.feature_names_in_)
        actual = set(processed_input.columns)

        print("MISSING COLUMNS:", expected - actual)
        print("EXTRA COLUMNS:", actual - expected)
        print("ACTUAL COLUMNS:", processed_input.columns.tolist())
        probabilities = model.predict_proba(processed_input)[:, 1]
        threshold = 0.65
        prediction = 1 if probabilities[0] >= threshold else 0

        st.subheader("📊 Loan Assessment Result")
        if prediction == 1:
            st.error(
                f"⚠️ High Risk / Potential Default (NPA) Detected! (Default "
                f"Probability: {probabilities[0]*100:.2f}%)"
            )
        else:
            st.success(
                f"✅ Low Risk! Loan Application Approved / Safe Zone. (Default "
                f"Probability: {probabilities[0]*100:.2f}%)"
            )


# ==================== TAB 2: BULK CSV BATCH AUDIT ====================
# ==================== TAB 2: BULK CSV BATCH AUDIT ====================
with tab2:
    st.header("📁 Bulk Portfolio Audit & Accuracy Evaluation")
    uploaded_file = st.file_uploader(
        "Upload Loan Portfolio CSV File for Batch Prediction", type=["csv"]
    )

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        if "Unnamed: 0" in batch_df.columns:
            batch_df = batch_df.drop(columns=["Unnamed: 0"])

        st.info(f"Loaded loan portfolio dataset with **{len(batch_df)}** applications.")

        if st.button("Run Portfolio Risk Audit"):
            target_col = None
            for col in batch_df.columns:
                cleaned_name = col.strip().lower().replace(" ", "_")
                if cleaned_name in ["loan_paid_back", "target", "default", "npa_status"]:
                    target_col = col
                    break

            if target_col:
                batch_df_features = batch_df.drop(columns=[target_col])
            else:
                batch_df_features = batch_df

            processed_batch = engineer_features(batch_df_features)

            batch_probs = model.predict_proba(processed_batch)[:, 1]
            batch_df["Default_Probability_%"] = (batch_probs * 100).round(2)
            batch_df["Predicted_NPA_Status"] = (batch_probs >= 0.65).astype(int)

            total_loans = len(batch_df)
            high_risk_loans = (batch_df["Predicted_NPA_Status"] == 1).sum()
            safe_loans = (batch_df["Predicted_NPA_Status"] == 0).sum()

            if target_col:
                batch_df["Result_Check"] = "❌ Incorrect"
                batch_df.loc[batch_df["Predicted_NPA_Status"] == batch_df[target_col], "Result_Check"] = "✅ Correct"
                
                correct_count = (batch_df["Predicted_NPA_Status"] == batch_df[target_col]).sum()
                wrong_count = total_loans - correct_count

            st.markdown("---")
            st.subheader("📊 Portfolio Risk Summary")

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Applications Audited", total_loans)
            col2.metric(
                "⚠️ High Risk / Defaulters",
                high_risk_loans,
                delta_color="inverse",
            )
            col3.metric("✅ Safe Loans Approved", safe_loans, delta_color="normal")
            
            # Agar actual target column mil gaya, toh Confusion Matrix (FP, FN, Accuracy) dikhao
            if target_col:
                st.markdown("---")
                st.subheader("🎯 Detailed Error Analysis (FP & FN Breakdown)")
                
                from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
                actuals = batch_df[target_col].values
                preds = batch_df["Predicted_NPA_Status"].values
                
                tn, fp, fn, tp = confusion_matrix(actuals, preds).ravel()
                
                acc = accuracy_score(actuals, preds) * 100
                prec = precision_score(actuals, preds, zero_division=0) * 100
                
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("✅ Correct", correct_count)
                m2.metric("❌ Wrong", wrong_count, delta_color="inverse")
                m3.metric("⚠️ False Positives (FP)", fp, help="Model ne safe loan ko High Risk bol diya")
                m4.metric("🚨 False Negatives (FN)", fn, help="Model ne Defaulter ko Safe bol diya")
                m5.metric("Model Accuracy", f"{acc:.2f}%")

            st.markdown("---")
            st.success("Batch portfolio processing complete!")
            st.dataframe(batch_df, use_container_width=True)

            csv_output = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Batch Audit Report (CSV)",
                csv_output,
                "Loan_Portfolio_Audit_Results.csv",
                "text/csv",
            )