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

# ==========================
    # 🔹 Sample Prediction Table
    # ==========================
    st.markdown("---")
    st.subheader("🔮 Sample Hasil Prediksi Promosi")

    # Contoh data prediksi (mock)
    sample_data = {
        "Employee_ID": [f"EMP{i:04d}" for i in range(1, 11)],
        "prediction": ["No", "Yes", "Yes", "No", "No", "No", "No", "Yes", "No", "Yes"],
        "probability": [0.052, 0.161, 0.347, 0.181, 0.197, 0.106, 0.582, 0.261, 0.552, 0.040],
        "confidence": ["High", "High", "Medium", "High", "High", "High", "Medium", "High", "Medium", "High"],
        "recommendation": ["Not Ready", "Promote", "Promote", "Not Ready", "Not Ready", 
                           "Not Ready", "Not Ready", "Promote", "Not Ready", "Promote"]
    }
    pred_df = pd.DataFrame(sample_data)

    # Styling rekomendasi
    def highlight_recommendation(val):
        color = "#B2F0B2" if val == "Promote" else "#F8B8B8"
        return f"background-color: {color}"

    styled_pred = pred_df.style.applymap(highlight_recommendation, subset=["recommendation"]).format({"probability": "{:.3f}"})
    st.dataframe(styled_pred, use_container_width=True)

    # Ringkasan kecil di bawah tabel
    colA, colB = st.columns(2)
    with colA:
        total_promote = (pred_df["recommendation"] == "Promote").sum()
        st.metric("🏅 Total Rekomendasi Promote", total_promote)
    with colB:
        avg_prob = pred_df["probability"].mean()
        st.metric("📈 Rata-rata Probability", f"{avg_prob:.2f}")


# Jalankan halaman dashboard
show_dashboard()
