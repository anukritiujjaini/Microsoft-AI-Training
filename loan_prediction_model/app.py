"""
app.py
------
Streamlit dashboard for the Loan Approval Prediction project.
Designed to be simple enough to demo live in front of a non-technical
audience: fill in the form on the left, click Predict, see the result.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Loan Approval Predictor ",
    page_icon="🏦",
    layout="centered",
)



# ---------------------------------------------------------------------
# LOAD MODEL + SUPPORTING FILES (cached so it only loads once)
# ---------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("loan_model.pkl")
    encoders = joblib.load("label_encoders.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    meta = joblib.load("model_meta.pkl")
    return model, encoders, feature_cols, meta

model, encoders, feature_cols, meta = load_artifacts()

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>

/* Hide Streamlit menu */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Background */
.stApp{
    background: linear-gradient(135deg,#0f172a,#1e293b);
}

/* Main container */
.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

/* Title */
h1{
    color:white !important;
    text-align:center;
    font-weight:700;
}

/* Caption */
[data-testid="stCaptionContainer"]{
    text-align:center;
    color:#cbd5e1;
}

/* Cards (containers) */
div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stHorizontalBlock"]){
    background:#1e293b;
    padding:20px;
    border-radius:15px;
    border:1px solid #334155;
}

/* Labels */
label{
    color:#f8fafc !important;
    font-weight:600 !important;
}

/* Metrics */
[data-testid="metric-container"]{
    background:#1e293b;
    border:1px solid #334155;
    border-radius:15px;
    padding:15px;
}

/* Button */
.stButton>button{
    width:100%;
    height:55px;
    border-radius:12px;
    border:none;
    background:linear-gradient(90deg,#2563eb,#3b82f6);
    color:white;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:linear-gradient(90deg,#1d4ed8,#2563eb);
}

/* Success */
div[data-testid="stSuccess"]{
    border-radius:12px;
    border-left:6px solid #22c55e;
}

/* Error */
div[data-testid="stError"]{
    border-radius:12px;
    border-left:6px solid #ef4444;
}

/* Progress */
.stProgress > div > div > div > div{
    background:#3b82f6;
}

/* Divider */
hr{
    border:none;
    height:1px;
    background:#334155;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------
st.markdown("""
<h1>🏦 Loan Approval Prediction System</h1>
<p style='text-align:center;color:#cbd5e1;font-size:18px;'>
Machine Learning Based Loan Eligibility Assessment
</p>
<p style='text-align:center;color:#94a3b8;'>
Developed by <b>Anukriti Ujjainiya</b>
</p>
""", unsafe_allow_html=True)

c1,c2,c3 = st.columns(3)

with c1:
    st.metric("🤖 Model", meta["model_name"])

with c2:
    st.metric("🎯 Accuracy", f"{meta['accuracy']*100:.1f}%")

with c3:
    st.metric("📊 Dataset", "Kaggle Loan")

st.divider()

# ---------------------------------------------------------------------
# INPUT FORM
# ---------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

with col2:
    applicant_income = st.number_input("Applicant Monthly Income (₹)", min_value=0, value=5000, step=500)
    coapplicant_income = st.number_input("Co-applicant Monthly Income (₹)", min_value=0, value=0, step=500)
    loan_amount = st.number_input("Loan Amount (in thousands ₹)", min_value=0, value=120, step=10)
    loan_term = st.selectbox("Loan Term (days)", [360, 180, 120, 84, 60, 36, 12])
    credit_history = st.selectbox("Has Good Credit History?", ["Yes", "No"])

st.divider()

# ---------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------
if st.button("🔍 Predict Loan Approval", use_container_width=True, type="primary"):

    # Build a single-row dataframe matching training-time preprocessing
    raw_input = {
        "Gender": gender,
        "Married": married,
        "Dependents": 3 if dependents == "3+" else int(dependents),
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": 1 if credit_history == "Yes" else 0,
        "Property_Area": property_area,
    }

    row = pd.DataFrame([raw_input])

    # Apply the SAME label encoders used during training
    for col in ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]:
        row[col] = row[col].astype(str)  # match the string encoding used in train.py
        row[col] = encoders[col].transform(row[col])

    # Ensure column order matches training
    row = row[feature_cols]

    prediction = model.predict(row)[0]
    probability = model.predict_proba(row)[0][1]  # probability of "Approved"

    st.markdown("## 📋 Prediction Result")
    if prediction == 1:
        st.success(f"🎉 Loan Approved\n\nConfidence : {probability*100:.1f}%")
    else:
        st.error(f"❌ Loan Rejected\n\nConfidence : {(1-probability)*100:.1f}%")

    st.progress(float(probability))
    st.caption("This is a prediction from a trained ML model, not a bank decision.")

st.divider()
st.markdown("""
<hr>
<p style='text-align:center;color:#94a3b8'>
🏦 Loan Approval Prediction Dashboard
<br><br>
Python • Streamlit • Scikit-Learn • Pandas
<br><br>
Designed & Developed by <b>Anukriti Ujjainiya</b>
</p>
""", unsafe_allow_html=True)
