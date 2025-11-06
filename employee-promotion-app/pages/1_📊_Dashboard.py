import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle

# =============================
# PAGE CONFIGURATION
# =============================
st.set_page_config(page_title="📊 Employee Dashboard", layout="wide")

# =============================
# LOAD MODEL & DATA
# =============================

@st.cache_resource
def load_model():
    """Load trained model and feature columns."""
    base_path = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_path, "models", "rf_model.pkl")
    feature_path = os.path.join(base_path, "models", "feature_columns.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(feature_path, "rb") as f:
        feature_columns = pickle.load(f)

    return model, feature_columns


@st.cache_data
def load_data():
    """Load cleaned dataset."""
    base_path = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_path, "data", "dataset_cleaning.csv")
    df = pd.read_csv(data_path)
    return df


# =============================
# MAIN DASHBOARD FUNCTION
# =============================

def show_dashboard():
    st.title("🏢 Employee Promotion Dashboard")

    # Load data & model
    df = load_data()
    model, feature_columns = load_model()

    # Pastikan fitur sesuai
    X = df[feature_columns]

    # Prediksi dari model
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]  # Probabilitas kelas "Promote"

    # Gabungkan hasil ke data
    df["Predicted_Promotion"] = predictions
    df["Promotion_Probability"] = probabilities

    # =============================
    # METRICS CARD
    # =============================
    total_employee = len(df)
    predicted_promotions = df["Predicted_Promotion"].sum()
    promotion_rate = (predicted_promotions / total_employee) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("👥 Total Employees", f"{total_employee:,}")
    col2.metric("🎯 Predicted Promotions", f"{predicted_promotions:,}")
    col3.metric("📈 Promotion Rate", f"{promotion_rate:.2f}%")

    st.markdown("---")

    # =============================
    # VISUALISASI 1: Distribusi Prediksi
    # =============================
    st.subheader("📊 Distribusi Prediksi Promosi")
    fig1, ax1 = plt.subplots()
    df["Predicted_Promotion"].value_counts().plot(
        kind="bar",
        ax=ax1,
        color=["#FFB74D", "#4CAF50"],
        edgecolor="black"
    )
    ax1.set_xlabel("Predicted Promotion (0=No, 1=Yes)")
    ax1.set_ylabel("Jumlah Karyawan")
    ax1.set_title("Distribusi Hasil Prediksi Promosi")
    st.pyplot(fig1)

    # =============================
    # VISUALISASI 2: Rata-rata skor performa per level
    # =============================
    st.subheader("⭐ Rata-rata Performance Score per Position Level")
    avg_score = (
        df.groupby("Current_Position_Level")["Performance_Score"]
        .mean()
        .sort_values(ascending=False)
    )
    st.bar_chart(avg_score)

    # =============================
    # TABEL CONTOH DATA
    # =============================
    st.subheader("📋 Contoh Data Karyawan & Prediksi")
    st.dataframe(
        df[[
            "Employee_ID", "Age", "Years_at_Company", "Performance_Score",
            "Leadership_Score", "Current_Position_Level",
            "Predicted_Promotion", "Promotion_Probability"
        ]].head(10),
        hide_index=True
    )

# Jalankan halaman
show_dashboard()
