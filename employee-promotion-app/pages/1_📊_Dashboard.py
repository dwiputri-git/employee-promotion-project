import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

st.set_page_config(page_title="📊 Employee Dashboard", layout="wide")

@st.cache_resource
def load_model():
    """Load trained Random Forest model and feature columns from ROOT folder."""
    # Path file langsung dari root project
    model_path = os.path.join(os.path.dirname(__file__), "..", "rf_model.pkl")
    feature_path = os.path.join(os.path.dirname(__file__), "..", "feature_columns.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(feature_path, "rb") as f:
        feature_columns = pickle.load(f)

    return model, feature_columns


@st.cache_data
def load_data():
    """Load cleaned dataset."""
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_cleaning.csv")
    df = pd.read_csv(data_path)
    return df


def show_dashboard():
    st.title("🏢 Employee Promotion Dashboard")

    # Load data & model
    df = load_data()
    model, feature_columns = load_model()

    # Prediksi
    X = df[feature_columns]
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    df["Predicted_Promotion"] = predictions
    df["Promotion_Probability"] = probabilities

    # Metric Cards
    total_employee = len(df)
    predicted_promotions = df["Predicted_Promotion"].sum()
    promotion_rate = (predicted_promotions / total_employee) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("👥 Total Employees", f"{total_employee:,}")
    col2.metric("🎯 Predicted Promotions", f"{predicted_promotions:,}")
    col3.metric("📈 Promotion Rate", f"{promotion_rate:.2f}%")

    st.markdown("---")

    # Distribusi Prediksi
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

    # Rata-rata Skor Per Level
    st.subheader("⭐ Rata-rata Performance Score per Position Level")
    avg_score = (
        df.groupby("Current_Position_Level")["Performance_Score"]
        .mean()
        .sort_values(ascending=False)
    )
    st.bar_chart(avg_score)

    # Contoh Data
    st.subheader("📋 Contoh Data & Hasil Prediksi")
    st.dataframe(
        df[[
            "Employee_ID", "Age", "Years_at_Company", "Performance_Score",
            "Leadership_Score", "Current_Position_Level",
            "Predicted_Promotion", "Promotion_Probability"
        ]].head(10),
        hide_index=True
    )

show_dashboard()
