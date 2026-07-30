"""
train_model.py
----------------
Trains a Decision Tree and a Random Forest on the Kaggle "Loan Prediction
Problem Dataset" and saves the best-performing model to disk for the
Streamlit app to load.

BEFORE RUNNING:
1. Download the dataset from Kaggle:
   https://www.kaggle.com/datasets/ninzaami/loan-predication
   (or any "Loan Prediction Problem Dataset" - they all share the same
   standard columns listed below)
2. Save it as: loan_data.csv  (in the same folder as this script)

Expected columns (standard for this dataset):
Loan_ID, Gender, Married, Dependents, Education, Self_Employed,
ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
Credit_History, Property_Area, Loan_Status
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ---------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------
df = pd.read_csv("loan_data.csv")
print(f"Loaded {len(df)} rows.")

# Drop the ID column - it has no predictive value
if "Loan_ID" in df.columns:
    df = df.drop(columns=["Loan_ID"])

# ---------------------------------------------------------------------
# 2. CLEAN / IMPUTE MISSING VALUES
# ---------------------------------------------------------------------
# Categorical columns -> fill with mode
cat_cols = ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History"]
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

# Numeric columns -> fill with median
num_cols = ["LoanAmount", "Loan_Amount_Term"]
for col in num_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# ---------------------------------------------------------------------
# 3. ENCODE CATEGORICAL VARIABLES
# ---------------------------------------------------------------------
# Dependents has a "3+" category -> convert to numeric
df["Dependents"] = df["Dependents"].replace("3+", 3).astype(float)

label_encoders = {}
categorical_to_encode = ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]

for col in categorical_to_encode:
    df[col] = df[col].astype(str)  # force text, in case the CSV already has numeric codes
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # save so the Streamlit app can use the same mapping

# Target column
target_le = LabelEncoder()
df["Loan_Status"] = target_le.fit_transform(df["Loan_Status"])  # Y=1, N=0
label_encoders["Loan_Status"] = target_le

# ---------------------------------------------------------------------
# 4. TRAIN / TEST SPLIT
# ---------------------------------------------------------------------
X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

FEATURE_COLUMNS = list(X.columns)
print("Feature columns:", FEATURE_COLUMNS)

# ---------------------------------------------------------------------
# 5. TRAIN DECISION TREE (baseline)
# ---------------------------------------------------------------------
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_preds)
dt_f1 = f1_score(y_test, dt_preds)

print("\n--- Decision Tree ---")
print(f"Accuracy: {dt_acc:.4f}  |  F1 Score: {dt_f1:.4f}")

# ---------------------------------------------------------------------
# 6. TRAIN RANDOM FOREST
# ---------------------------------------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=8, random_state=42
)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
rf_f1 = f1_score(y_test, rf_preds)

print("\n--- Random Forest ---")
print(f"Accuracy: {rf_acc:.4f}  |  F1 Score: {rf_f1:.4f}")
print(classification_report(y_test, rf_preds, target_names=["Rejected", "Approved"]))

# ---------------------------------------------------------------------
# 7. PICK THE BEST MODEL AND SAVE EVERYTHING THE APP NEEDS
# ---------------------------------------------------------------------
if rf_acc >= dt_acc:
    best_model, best_name = rf_model, "Random Forest"
else:
    best_model, best_name = dt_model, "Decision Tree"

print(f"\nBest model: {best_name}")

joblib.dump(best_model, "loan_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(FEATURE_COLUMNS, "feature_columns.pkl")
joblib.dump(
    {"model_name": best_name, "accuracy": max(rf_acc, dt_acc), "f1": max(rf_f1, dt_f1)},
    "model_meta.pkl",
)

print("\nSaved: loan_model.pkl, label_encoders.pkl, feature_columns.pkl, model_meta.pkl")
print("You can now run: streamlit run app.py")
