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
    MODEL_PATH = os.path.join(BASE_DIR, "..", "rf_model.pkl")
    FEATURE_PATH = os.path.join(BASE_DIR, "..", "feature_columns.pkl")

    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(FEATURE_PATH, "rb") as f:
            feature_columns = pickle.load(f)
        return model, feature_columns
    except FileNotFoundError:
        st.error("❌ File model tidak ditemukan di root folder repository.")
        st.stop()

model, feature_columns = load_model()

# -----------------------------
# ✅ Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "..", "data", "dataset_cleaning.csv")
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

# -----------------------------
# ✅ Feature Engineering
# -----------------------------
def feature_engineering(df):
    df = df.copy()

    # Tambahkan fitur baru (pastikan nama sesuai model)
    if 'Training_Hours' in df.columns:
        df['Training_Level'] = pd.cut(
            df['Training_Hours'],
            bins=[0, 20, 50, 100, 200],
            labels=['Low', 'Medium', 'High', 'Intense'],
            include_lowest=True
        )
    
    if 'Projects_Handled' in df.columns and 'Years_at_Company' in df.columns:
        df['Projects_per_Years'] = df['Projects_Handled'] / df['Years_at_Company'].replace(0, np.nan)
        # Kalau skew → log transform
        if df['Projects_per_Years'].skew() > 1:
            df['Projects_per_Years_log'] = np.log1p(df['Projects_per_Years'])
        else:
            df['Projects_per_Years_log'] = df['Projects_per_Years']
    
    # Jika kolom lain yang diperlukan model tidak ada, isi dengan 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    return df

df = feature_engineering(df)

# -----------------------------
# ✅ Generate Predictions
# -----------------------------
@st.cache_data
def generate_predictions(df):
    X = df[feature_columns].copy()
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    df["Prediction"] = preds
    df["Probability"] = probs
    return df

df_pred = generate_predictions(df)

# -----------------------------
# ✅ Dashboard Layout
# -----------------------------
st.title("📊 Employee Promotion Dashboard")

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
# ✅ Sample Data
# -----------------------------
st.subheader("📋 Sample Data dengan Prediksi")
st.dataframe(df_pred.head(10))
