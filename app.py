import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Bank Marketing Prediction")

st.title("Portuguese Bank Marketing Prediction")
st.write("Predict whether a customer will subscribe to a term deposit")

# Load dataset
df = pd.read_csv("bank-full.csv", sep=";")

# Encode target
df["y"] = df["y"].map({"yes": 1, "no": 0})

X = df.drop("y", axis=1)
y = df["y"]

cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]
)

model.fit(X, y)

st.subheader("Enter Customer Details")

age = st.slider("Age", 18, 95, 30)
balance = st.number_input("Account Balance", -5000, 100000, 1000)
housing = st.selectbox("Housing Loan", ["yes", "no"])
loan = st.selectbox("Personal Loan", ["yes", "no"])
campaign = st.slider("Campaign Contacts", 1, 50, 2)
previous = st.slider("Previous Contacts", 0, 50, 0)

input_df = pd.DataFrame(
    {
        "age": [age],
        "balance": [balance],
        "housing": [housing],
        "loan": [loan],
        "campaign": [campaign],
        "previous": [previous],
    }
)

# Fill missing columns safely
for col in X.columns:
    if col not in input_df.columns:
        input_df[col] = X[col].mode()[0]

input_df = input_df[X.columns]

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.success(f"✅ Customer WILL subscribe (Probability: {prob:.2%})")
    else:
        st.error(f"❌ Customer will NOT subscribe (Probability: {prob:.2%})")
