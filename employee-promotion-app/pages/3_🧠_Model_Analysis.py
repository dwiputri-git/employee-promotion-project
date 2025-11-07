import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import plotly.express as px

st.set_page_config(page_title="Model Analysis", layout="wide")

# --- Load Model & Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("employee-promotion-app/data/cleaned_data.csv")
    return df

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


# --- Fungsi Visualisasi ---
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Not Promoted (0)", "Promoted (1)"]

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        linewidths=1,
        linecolor="white"
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=13, pad=10)
    st.pyplot(fig)


def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
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
    Halaman ini menampilkan **evaluasi performa model Random Forest (rf_model2.pkl)**  
    yang digunakan untuk memprediksi kelayakan promosi karyawan.
    """)

    df = load_data()
    model, feature_columns = load_model()

    if "Promotion_Eligible" not in df.columns:
        st.error("Kolom 'Promotion_Eligible' tidak ditemukan di dataset.")
        return

    X = df[feature_columns]
    y_true = df["Promotion_Eligible"]

    # Coba prediksi probabilitas (kalau ada)
    try:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
    except Exception:
        y_pred = model.predict(X)
        y_prob = [0.5] * len(y_pred)  # fallback

    # --- Layout 2 kolom atas ---
    col1, col2 = st.columns(2)

    # --- Confusion Matrix ---
    with col1:
        st.subheader("📉 Confusion Matrix")
        plot_confusion_matrix(y_true, y_pred)

    # --- ROC Curve ---
    with col2:
        st.subheader("📈 ROC Curve")
        plot_roc_curve(y_true, y_prob)

    # --- Classification Report ---
    st.subheader("📊 Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(
        report_df.style.background_gradient(cmap="Oranges"),
        use_container_width=True
    )

    # --- Feature Importance ---
    st.subheader("🏗️ Feature Importance")

    # Ambil step RF kalau model pipeline
    rf_step = None
    if hasattr(model, "named_steps"):
        for name, step in model.named_steps.items():
            if hasattr(step, "feature_importances_"):
                rf_step = step
                break
    if rf_step is None and hasattr(model, "feature_importances_"):
        rf_step = model

    if rf_step is not None:
        importances = rf_step.feature_importances_
        feature_importances = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        fig = px.bar(
            feature_importances.head(10).sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Oranges",
            title="Top 10 Most Important Features"
        )
        fig.update_layout(
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            title_font=dict(size=16),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Feature importance tidak tersedia pada model ini.")

    # --- Catatan Analisis ---
    st.markdown("""
    **Catatan:**
    - Warna oranye menandakan jumlah prediksi pada masing-masing kategori.
    - ROC Curve menggambarkan kemampuan model dalam membedakan dua kelas.
    - Feature Importance menunjukkan fitur yang paling berpengaruh terhadap keputusan promosi.
    """)


# --- Run Page ---
show_model_analysis()
