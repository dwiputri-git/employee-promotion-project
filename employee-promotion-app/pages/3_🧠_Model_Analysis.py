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
    df = pd.read_csv("employee-promotion-app/data/cleaned_data_with_target.csv")
    return df

@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "..", "rf_model.pkl")
    FEATURE_PATH = os.path.join(BASE_DIR, "..", "feature_columns.pkl")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEATURE_PATH, "rb") as f:
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

    # --- Feature Importance & SHAP Analysis ---
    st.subheader("🏗️ Feature Importance & Explainability")

    try:
        import plotly.express as px
        import shap
        import numpy as np

        # Cari langkah dalam pipeline yang punya feature_importances_
        rf_step = None
        if hasattr(model, "named_steps"):
            for name, step in model.named_steps.items():
                if hasattr(step, "feature_importances_"):
                    rf_step = step
                    st.caption(f"✅ Feature importances ditemukan di langkah: '{name}'")
                    break

        if rf_step is None and hasattr(model, "feature_importances_"):
            rf_step = model
        if rf_step is None:
            raise AttributeError("Tidak ada langkah dalam pipeline yang memiliki feature_importances_.")

        # --- Feature Importance ---
        importances = rf_step.feature_importances_
        n = min(len(importances), len(feature_columns))
        feature_importances = pd.DataFrame({
            "Feature": feature_columns[:n],
            "Importance": importances[:n]
        }).sort_values(by="Importance", ascending=False)

        top_features = feature_importances.head(10)
        fig = px.bar(
            top_features.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues",
            title="Top 10 Most Important Features (Random Forest)",
            text=top_features["Importance"].apply(lambda x: f"{x:.3f}")
        )
        fig.update_layout(
            title_font=dict(size=18),
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        # --- SHAP Analysis ---
        st.markdown("### 🔍 SHAP Value Analysis")
        st.info("""
        SHAP (SHapley Additive exPlanations) membantu menjelaskan **bagaimana setiap fitur memengaruhi prediksi model**.
        Warna merah menunjukkan fitur yang meningkatkan kemungkinan promosi, biru menurunkan.
        """)

        # Ambil subset kecil untuk efisiensi
        X_sample = X.sample(n=min(200, len(X)), random_state=42)

        # --- Pastikan semua fitur numerik ---
        # Jika ada kolom kategorikal, lakukan one-hot encoding
        X_encoded = pd.get_dummies(X_sample)

        # Pastikan kolom cocok dengan yang digunakan model (hanya kolom yang dikenal)
        X_encoded = X_encoded.reindex(columns=[c for c in X_encoded.columns if c in X_encoded.columns], fill_value=0)

        # Buat TreeExplainer dan hitung shap values
        explainer = shap.TreeExplainer(rf_step)
        shap_values = explainer.shap_values(X_encoded)

        # --- Global SHAP Summary Plot ---
        st.subheader("🌐 SHAP Summary Plot")
        shap.summary_plot(shap_values[1], X_encoded, plot_type="dot", show=False)
        st.pyplot(bbox_inches="tight")

        # --- Global Mean Impact (Bar Plot) ---
        st.subheader("🏆 SHAP Feature Impact (Global Importance)")
        shap.summary_plot(shap_values[1], X_encoded, plot_type="bar", show=False)
        st.pyplot(bbox_inches="tight")

        # --- Local Explanation untuk 1 Employee ---
        st.subheader("👤 SHAP Explanation per Employee")
        emp_id = st.selectbox("Pilih Employee_ID untuk analisis detail:", df["Employee_ID"].unique())

        emp_data = df[df["Employee_ID"] == emp_id][feature_columns]
        emp_encoded = pd.get_dummies(emp_data)
        emp_encoded = emp_encoded.reindex(columns=X_encoded.columns, fill_value=0)

        shap.force_plot(
            explainer.expected_value[1],
            explainer.shap_values(emp_encoded)[1],
            emp_encoded,
            matplotlib=True,
            show=False
        )
        st.pyplot(bbox_inches="tight")

    except Exception as e:
        st.warning(f"Feature importance atau SHAP tidak dapat ditampilkan: {e}")



    # --- Catatan Tambahan ---
    st.markdown("""
    **Catatan Analisis:**
    - Model dengan AUC tinggi berarti lebih baik membedakan antara karyawan yang layak dan tidak layak promosi.  
    - Jika *Precision* rendah, artinya ada risiko merekomendasikan promosi untuk karyawan yang belum siap.  
    - HR dapat menyesuaikan threshold promosi berdasarkan kebutuhan bisnis.
    """)


show_model_analysis()
