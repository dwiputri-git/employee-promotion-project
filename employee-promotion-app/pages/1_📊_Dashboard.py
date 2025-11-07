import os
import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Dashboard", layout="wide")

# -----------------------------
# ✅ Load Model & Feature Columns
# -----------------------------
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "..", "rf_model2.pkl")
    FEATURE_PATH = os.path.join(BASE_DIR, "..", "feature_columns2.pkl")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEATURE_PATH, "rb") as f:
        feature_columns = pickle.load(f)
    return model, feature_columns

model, feature_columns = load_model()

# -----------------------------
# ✅ Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "..", "data", "cleaned_data.csv")
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

# -----------------------------
# ✅ Generate Predictions (Simple version)
# -----------------------------
@st.cache_data
def generate_predictions(df):
    X = df[feature_columns].copy()
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    df = df.copy()
    df["Prediction"] = preds
    df["Probability"] = probs

    # --- Final Recommendation Logic ---
    def final_recommendation(p):
        if p >= 0.6:
            return "Promote"
        elif p >= 0.5:
            return "Need Review"
        else:
            return "Not Ready"

    df["Recommendation"] = df["Probability"].apply(final_recommendation)
    return df

df_pred = generate_predictions(df)

# -----------------------------
# ✅ Dashboard Layout
# -----------------------------
st.title("📊 Dashboard")

col1, col2, col3 = st.columns(3)
total_employees = len(df_pred)
predicted_promotions = (df_pred["Recommendation"] == "Promote").sum()
promotion_rate = (predicted_promotions / total_employees) * 100

col1.metric("👥 Total Employees", total_employees)
col2.metric("🏅 Predicted Promotions", predicted_promotions)
col3.metric("📈 Promotion Rate", f"{promotion_rate:.2f}%")

st.divider()

# -----------------------------
# ✅ Visualisasi
# -----------------------------
st.subheader("Distribusi Prediksi Promosi")
fig, ax = plt.subplots()
df_pred["Recommendation"].value_counts().plot(
    kind="bar", color=["#FFCDD2", "#FFF59D", "#C8E6C9"], ax=ax
)
plt.title("Distribusi Hasil Rekomendasi")
plt.xticks(rotation=0)
st.pyplot(fig)

st.subheader("Rata-rata Performance Score per Level Jabatan")
avg_score = df_pred.groupby("Current_Position_Level")["Performance_Score"].mean().sort_values(ascending=False)
st.bar_chart(avg_score)

# -----------------------------
# ✅ Sample Data
# -----------------------------
st.subheader("📋 Sample Data dengan Prediksi")
df_pred = generate_predictions(df)

# Format angka biar rapi
numeric_cols = df_pred.select_dtypes(include=["float", "int"]).columns
exclude_cols = ["Probability"]
for col in numeric_cols:
    if col not in exclude_cols:
        df_pred[col] = df_pred[col].round(0).astype("Int64")

# Pilih kolom penting aja
selected_cols = [
    'Employee_ID', 
    'Age', 
    'Years_at_Company', 
    'Performance_Score',
    'Leadership_Score', 
    'Training_Hours', 
    'Projects_Handled',
    'Peer_Review_Score', 
    'Current_Position_Level', 
    'Training_Level', 
    'Leadership_Level', 
    'Projects_per_Years',
    'Project_Level', 
    'Tenure_Level', 
    'Age_Group',
    'Promotion_Eligible',
    "Prediction",
    "Probability",
    "Recommendation"
]

selected_cols = [col for col in selected_cols if col in df_pred.columns]
df_show = df_pred[selected_cols].copy()

# Styling warna
def highlight_recommendation(val):
    if val == "Promote":
        return "background-color: #C8E6C9; color: #1B5E20; font-weight: bold;"
    elif val == "Need Review":
        return "background-color: #FFF9C4; color: #E65100; font-weight: bold;"
    elif val == "Not Ready":
        return "background-color: #FFCDD2; color: #C62828; font-weight: bold;"
    return ""

st.dataframe(
    df_show.style
        .applymap(highlight_recommendation, subset=["Recommendation"])
        .format({"Probability": "{:.2%}"})
)
