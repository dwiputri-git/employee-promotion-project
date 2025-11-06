import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Employee Dashboard", layout="wide")

# ==========================
# 🔹 Load Data Function
# ==========================
@st.cache_data
def load_real_data():
    base_path = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_path, "data", "dataset_cleaning.csv")
    df = pd.read_csv(data_path)
    return df

# ==========================
# 📊 Dashboard Page
# ==========================
def show_dashboard():
    st.title("📊 Employee Promotion Dashboard")
    st.markdown("Selamat datang di dashboard HR! Berikut ringkasan data promosi karyawan berdasarkan hasil prediksi dan performa internal.")

    # Load data
    df = load_real_data()

    # ==========================
    # 🔹 Summary Metrics
    # ==========================
    total_employee = len(df)
    total_promoted = df['Promotion_Eligible'].sum() if 'Promotion_Eligible' in df.columns else 0
    promotion_rate = (total_promoted / total_employee * 100) if total_employee > 0 else 0

    st.markdown("### 📈 Ringkasan Utama")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("👥 Total Employees", f"{total_employee:,}")
    with col2:
        st.metric("🏅 Predicted Promotions", f"{total_promoted:,}")
    with col3:
        st.metric("📊 Promotion Rate", f"{promotion_rate:.2f}%")

    # ==========================
    # 🔹 Data Preview
    # ==========================
    with st.expander("🔍 Lihat Sample Data"):
        st.dataframe(df.head())

    # ==========================
    # 🔹 Visualisasi Target
    # ==========================
    st.markdown("---")
    st.subheader("🎯 Distribusi Target (Promotion Eligibility)")

    fig, ax = plt.subplots()
    df['Promotion_Eligible'].value_counts().plot(kind='bar', ax=ax, color=['skyblue', 'lightgreen'])
    ax.set_xlabel("Promotion Eligible (0 = No, 1 = Yes)")
    ax.set_ylabel("Jumlah Karyawan")
    ax.set_title("Distribusi Karyawan Berdasarkan Status Promosi")
    st.pyplot(fig)

    # ==========================
    # 🔹 Visualisasi Rata-rata per Level
    # ==========================
    st.markdown("---")
    st.subheader("📊 Rata-rata Performance Score per Position Level")

    avg_score = df.groupby('Current_Position_Level')['Performance_Score'].mean().sort_values(ascending=False)
    st.bar_chart(avg_score)

# Jalankan halaman dashboard
show_dashboard()
