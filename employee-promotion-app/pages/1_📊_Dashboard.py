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
# ✅ Generate Predictions (+ Confidence)
# -----------------------------
THRESHOLD = 0.4 

def prob_to_confidence_label(p: float, thr: float = THRESHOLD) -> str:
    dist = abs(p - thr)
    if dist >= 0.30:
        return "High"
    elif dist >= 0.10:
        return "Medium"
    else:
        return "Low"

@st.cache_data
def generate_predictions(df):
    X = df[feature_columns].copy()
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    # Kolom utama
    df = df.copy()
    df["Prediction"] = preds
    df["Probability"] = probs

    # --- Confidence label
    df["Confidence"] = df["Probability"].apply(prob_to_confidence_label)

    # --- Final Recommendation Logic
    def final_recommendation(row):
        p = row["Probability"]
        conf = row["Confidence"]
        if p >= 0.7 and conf == "High":
            return "Promote"
        elif 0.5 <= p < 0.7:
            return "Promote (Need Review)"
        elif p < 0.5 and conf == "Low":
            return "Not Ready"
        else:
            return "Not Ready"

    df["Recommendation"] = df.apply(final_recommendation, axis=1)
    return df

df_pred = generate_predictions(df)

# -----------------------------
# ✅ Dashboard Layout
# -----------------------------
st.title("📊 Dashboard")

col1, col2, col3 = st.columns(3)
total_employees = len(df_pred)
predicted_promotions = int(df_pred["Prediction"].sum())
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
df_pred["Prediction"].value_counts().plot(kind="bar", color=["#1f77b4", "#2ca02c"], ax=ax)
plt.xticks([0, 1], ["Not Promoted", "Promoted"], rotation=0)
plt.title("Distribusi Hasil Prediksi")
st.pyplot(fig)

st.subheader("Rata-rata Performance Score per Level Jabatan")
avg_score = df_pred.groupby("Current_Position_Level")["Performance_Score"].mean().sort_values(ascending=False)
st.bar_chart(avg_score)

# -----------------------------
# ✅ Sample Data (Custom Columns + Styling)
# -----------------------------
st.subheader("📋 Sample Data dengan Prediksi")
df_pred = generate_predictions(df)  # pastikan sudah include kolom baru

# format angka biar rapi
numeric_cols = df_pred.select_dtypes(include=["float", "int"]).columns
exclude_cols = ["Probability", "Promotion_Rate", "Confidence_Score"]
for col in numeric_cols:
    if col not in exclude_cols:
        df_pred[col] = df_pred[col].round(0).astype("Int64")

# kolom yang ditampilkan
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
    "Confidence",          # <-- NEW
    "Confidence_Score",    # <-- NEW (opsional buat diagnosa)
    "Recommendation"
]
selected_cols = [c for c in selected_cols if c in df_pred.columns]
df_show = df_pred[selected_cols].copy()

# styling rekomendasi & confidence
def highlight_recommendation(val):
    if val == "Promote":
        return "background-color: #C8E6C9; color: #2E7D32; font-weight: bold;"
    elif val == "Promote (Need Review)":
        return "background-color: #e0d904; color: #C62828; font-weight: bold;"
    elif val == "Not Ready":
        return "background-color: #FFCDD2; color: #C62828; font-weight: bold;"
    return ""

def highlight_confidence(val):
    if val == "High":
        return "background-color: #E8F5E9; color: #1B5E20;"
    if val == "Medium":
        return "background-color: #FFF8E1; color: #E65100;"
    if val == "Low":
        return "background-color: #FFEBEE; color: #B71C1C;"
    return ""

st.dataframe(
    df_show.style
        .applymap(highlight_recommendation, subset=["Recommendation"])
        .applymap(highlight_confidence, subset=["Confidence"])
        .format({"Probability": "{:.2%}", "Confidence_Score": "{:.3f}"})
)
