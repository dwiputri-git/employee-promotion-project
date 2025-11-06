import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

st.set_page_config(page_title="Model Analysis", layout="wide")

# --- Load Model & Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned_data.csv")
    return df

@st.cache_resource
def load_model():
    with open("rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    return model, feature_columns


# --- Fungsi Visualisasi ---
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)


def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    st.pyplot(fig)


# --- Main Page ---
def show_model_analysis():
    st.title("🧠 Model Analysis")
    st.markdown("""
    Halaman ini menampilkan **evaluasi performa model Random Forest** 
    yang digunakan untuk memprediksi kelayakan promosi karyawan.
    """)

    df = load_data()
    model, feature_columns = load_model()

    # Pastikan data memiliki kolom target
    if "Promotion_Eligible" not in df.columns:
        st.error("Kolom 'Promotion_Eligible' tidak ditemukan di dataset.")
        return

    # Prediksi model
    X = df[feature_columns]
    y_true = df["Promotion_Eligible"]
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # --- Metrics summary ---
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    st.subheader("📊 Classification Report")
    st.dataframe(report_df.style.background_gradient(cmap="Blues"), use_container_width=True)

    # --- Confusion Matrix ---
    st.subheader("📉 Confusion Matrix")
    plot_confusion_matrix(y_true, y_pred)

    # --- ROC Curve ---
    st.subheader("📈 ROC Curve")
    plot_roc_curve(y_true, y_prob)

    # --- Feature Importance ---
    st.subheader("🏗️ Feature Importance")
    try:
        feature_importances = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": model.named_steps["model"].feature_importances_
        }).sort_values(by="Importance", ascending=False)
        st.bar_chart(feature_importances.set_index("Feature"))
    except Exception:
        st.info("Feature importance tidak tersedia untuk pipeline ini.")

    # --- Catatan Tambahan ---
    st.markdown("""
    **Catatan Analisis:**
    - Model dengan AUC tinggi berarti lebih baik membedakan antara karyawan yang layak dan tidak layak promosi.  
    - Jika *Precision* rendah, artinya ada risiko merekomendasikan promosi untuk karyawan yang belum siap.  
    - HR dapat menyesuaikan threshold promosi berdasarkan kebutuhan bisnis.
    """)


show_model_analysis()
