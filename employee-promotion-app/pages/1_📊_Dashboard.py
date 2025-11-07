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
# ✅ Generate Predictions
# -----------------------------
@st.cache_data
def generate_predictions(df):
    X = df[feature_columns].copy()
    probs = model.predict_proba(X)[:, 1]

    df = df.copy()
    df["Probability"] = probs

    # --- Final Recommendation Logic ---
    def final_recommendation(p):
        if p >= 0.5:
            return "Promote"
        else:
            return "Not Ready"

    df["Recommendation"] = df["Probability"].apply(final_recommendation)
    return df

df_pred = generate_predictions(df)

# -----------------------------
# ✅ Dashboard Layout
# -----------------------------
st.title("📊 Employee Promotion Dashboard")

col1, col2, col3 = st.columns(3)
total_employees = len(df_pred)
predicted_promotions = (df_pred["Recommendation"] == "Promote").sum()
promotion_rate = (predicted_promotions / total_employees) * 100

col1.metric("👥 Total Employees", total_employees)
col2.metric("🏅 Predicted Promotions", predicted_promotions)
col3.metric("📈 Promotion Rate", f"{promotion_rate:.2f}%")

st.divider()

# ============================================================
# 📊 Row 1 - Distribusi & Promotion Rate by Position
# ============================================================
st.subheader("📦 Distribusi & Promotion Rate")

colA, colB = st.columns(2)

with colA:
    st.markdown("**Distribusi Rekomendasi Promosi**")
    fig1, ax1 = plt.subplots()
    order = ["Not Ready", "Promote"]
    df_pred["Recommendation"].value_counts().reindex(order).plot(
        kind="bar", color=["#FFCDD2", "#C8E6C9"], ax=ax1
    )
    plt.title("Distribusi Hasil Rekomendasi")
    plt.xticks(rotation=0)
    plt.ylabel("Jumlah Karyawan")
    st.pyplot(fig1)

with colB:
    st.markdown("**Promotion Rate by Current Position Level**")
    promotion_by_level = (
        df_pred.groupby("Current_Position_Level")["Recommendation"]
        .apply(lambda x: (x == "Promote").mean() * 100)
        .sort_values(ascending=False)
    )
    fig2, ax2 = plt.subplots()
    promotion_by_level.plot(kind="bar", color="#90CAF9", ax=ax2)
    plt.title("Promotion Rate per Level Jabatan")
    plt.ylabel("Promotion Rate (%)")
    plt.xlabel("Current Position Level")
    st.pyplot(fig2)

# ============================================================
# 📊 Row 2 - Performance Score & Project Level
# ============================================================
st.subheader("🎯 Performance & Project Analysis")

colC, colD = st.columns(2)

with colC:
    st.markdown("**Performance Score vs Promotion Probability**")
    fig3, ax3 = plt.subplots()
    ax3.scatter(df_pred["Performance_Score"], df_pred["Probability"], alpha=0.6)
    plt.title("Performance Score vs Promotion Probability")
    plt.xlabel("Performance Score")
    plt.ylabel("Promotion Probability")
    st.pyplot(fig3)

with colD:
    st.markdown("**Project Level vs Promotion Rate**")
    promotion_by_project = (
        df_pred.groupby("Project_Level")["Recommendation"]
        .apply(lambda x: (x == "Promote").mean() * 100)
        .sort_index()
    )
    fig4, ax4 = plt.subplots()
    promotion_by_project.plot(kind="bar", color="#A5D6A7", ax=ax4)
    plt.title("Promotion Rate berdasarkan Project Level")
    plt.ylabel("Promotion Rate (%)")
    plt.xlabel("Project Level")
    st.pyplot(fig4)

# ============================================================
# 📋 Sample Data Table
# ============================================================
st.divider()
st.subheader("📋 Historical Prediction")

# Format angka biar rapi
numeric_cols = df_pred.select_dtypes(include=["float", "int"]).columns
exclude_cols = ["Probability"]
for col in numeric_cols:
    if col not in exclude_cols:
        df_pred[col] = df_pred[col].round(0).astype("Int64")

selected_cols = [
    "Employee_ID",
    "Age",
    "Years_at_Company",
    "Performance_Score",
    "Leadership_Score",
    "Training_Hours",
    "Projects_Handled",
    "Peer_Review_Score",
    "Current_Position_Level",
    "Project_Level",
    "Tenure_Level",
    "Promotion_Eligible",
    "Probability",
    "Recommendation",
]
selected_cols = [c for c in selected_cols if c in df_pred.columns]
df_show = df_pred[selected_cols].copy()

# Warna rekomendasi
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
