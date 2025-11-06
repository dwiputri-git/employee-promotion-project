import streamlit as st
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Employee Dashboard", layout="wide")

# ==========================
# 🔹 Load Model dan Data
# ==========================
@st.cache_resource
def load_model():
    base_path = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_path, "rf_model.pkl")
    columns_path = os.path.join(base_path, "models", "feature_columns.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(columns_path, "rb") as f:
        feature_columns = pickle.load(f)

    return model, feature_columns

@st.cache_data
def load_real_data():
    base_path = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_path, "data", "dataset_cleaning.csv")
    df = pd.read_csv(data_path)
    return df

# ==========================
# 📊 Dashboard
# ==========================
def show_dashboard():
    st.title("📊 Employee Promotion Dashboard")
    st.markdown("Selamat datang di dashboard HR! Berikut ringkasan promosi karyawan berdasarkan hasil prediksi model Random Forest.")

    # Load model & data
    model, feature_columns = load_model()
    df = load_real_data()

    # Pastikan semua fitur sesuai urutan training
    X = df[feature_columns]

    # Prediksi dari model
    df['probability'] = model.predict_proba(X)[:, 1]
    df['prediction'] = model.predict(X)
    df['prediction'] = df['prediction'].map({1: "Yes", 0: "No"})
    df['recommendation'] = df['prediction'].map({"Yes": "Promote", "No": "Not Ready"})

    # Confidence (optional)
    df['confidence'] = df['probability'].apply(lambda p: "High" if p > 0.7 or p < 0.3 else "Medium")

    # ==========================
    # 🔹 Summary Metrics
    # ==========================
    total_employee = len(df)
    total_promoted = (df['prediction'] == "Yes").sum()
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
    # 🔹 Visualisasi
    # ==========================
    st.markdown("---")
    st.subheader("🎯 Distribusi Prediksi Promosi")
    fig, ax = plt.subplots()
    df['prediction'].value_counts().plot(kind='bar', ax=ax, color=['skyblue', 'lightgreen'])
    ax.set_xlabel("Prediction (0 = No, 1 = Yes)")
    ax.set_ylabel("Jumlah Karyawan")
    ax.set_title("Distribusi Hasil Prediksi Promosi")
    st.pyplot(fig)

    # ==========================
    # 🔹 Sample Prediction Table
    # ==========================
    st.markdown("---")
    st.subheader("🔮 Sample Hasil Prediksi dari Model")

    sample_df = df[['Employee_ID', 'prediction', 'probability', 'confidence', 'recommendation']].head(10)

    # Styling
    def highlight_recommendation(val):
        color = "#B2F0B2" if val == "Promote" else "#F8B8B8"
        return f"background-color: {color}"

    styled_pred = sample_df.style.applymap(highlight_recommendation, subset=["recommendation"]).format({"probability": "{:.3f}"})
    st.dataframe(styled_pred, use_container_width=True)

    # Ringkasan tambahan
    colA, colB = st.columns(2)
    with colA:
        st.metric("🏅 Total Rekomendasi Promote", (df["recommendation"] == "Promote").sum())
    with colB:
        st.metric("📈 Rata-rata Probability", f"{df['probability'].mean():.2f}")

# Jalankan dashboard
show_dashboard()
