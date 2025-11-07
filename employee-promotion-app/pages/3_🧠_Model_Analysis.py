import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="Model Analysis", layout="wide")

# ============================
# MAIN PAGE
# ============================
def show_model_analysis():
    st.title("🧠 Model Analysis")
    st.markdown("""
    Halaman ini menampilkan **evaluasi performa model Random Forest** 
    yang digunakan untuk memprediksi kelayakan promosi karyawan.
    """)

    # ============================
    # 📈 METRICS SUMMARY
    # ============================
    st.markdown("### 📊 Model Performance Summary")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    card_style = """
        background-color: #f5f9ff;
        padding: 10px 0;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0px 1px 2px rgba(0,0,0,0.05);
        border: 1px solid #e6efff;
    """

    metric_html = lambda title, value: f"""
        <div style='{card_style}'>
            <p style='font-size:13px; color:#333; margin-bottom:4px; font-weight:500;'>{title}</p>
            <p style='font-size:18px; color:#000; margin:0; font-weight:600;'>{value}</p>
        </div>
    """

    # kamu bisa ubah angkanya manual di sini
    col1.markdown(metric_html("Accuracy", "0.707"), unsafe_allow_html=True)
    col2.markdown(metric_html("Precision", "0.5"), unsafe_allow_html=True)
    col3.markdown(metric_html("Recall", "0.164"), unsafe_allow_html=True)
    col4.markdown(metric_html("F1-Score", "0.247"), unsafe_allow_html=True)
    col5.markdown(metric_html("ROC AUC", "0.506"), unsafe_allow_html=True)
    col6.markdown(metric_html("PR AUC", "0.347"), unsafe_allow_html=True)

    st.markdown("---")

    # ============================
    # 📉 CONFUSION MATRIX (MANUAL)
    # ============================
    st.subheader("📉 Confusion Matrix")

    # nilai manual (ganti sesuai hasil modelmu)
    cm_data = [[550, 30],
               [45, 120]]
    labels = ["Not Promoted", "Promoted"]

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm_data, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)

    st.markdown("---")

    # ============================
    # 📈 ROC CURVE (MANUAL)
    # ============================
    st.subheader("📈 ROC Curve")

    # contoh data ROC manual
    fpr = [0.0, 0.05, 0.10, 0.20, 0.40, 1.0]
    tpr = [0.0, 0.60, 0.80, 0.90, 0.95, 1.0]
    roc_auc = 0.960

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    st.pyplot(fig)

    st.markdown("---")

    # ============================
    # 🏗️ FEATURE IMPORTANCE (MANUAL)
    # ============================
    st.subheader("🏗️ Feature Importance")

    # kamu bisa ubah daftar fitur & nilainya
    feature_importances = pd.DataFrame({
        "Feature": [
            "Previous Year Rating", 
            "Length of Service", 
            "Training Score Average", 
            "Awards Won", 
            "KPIs Met >80%", 
            "Age", 
            "Department", 
            "Education Level", 
            "Gender", 
            "Recruitment Channel"
        ],
        "Importance": [0.22, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04, 0.03, 0.02]
    })

    fig = px.bar(
        feature_importances.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
        title="Top 10 Most Important Features",
        text=feature_importances["Importance"].apply(lambda x: f"{x:.2f}")
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

    # ============================
    # NOTES
    # ============================
    st.markdown("""
    **Catatan Analisis:**
    - Model dengan AUC tinggi berarti lebih baik membedakan antara karyawan yang layak dan tidak layak promosi.  
    - Jika *Precision* rendah, artinya ada risiko merekomendasikan promosi untuk karyawan yang belum siap.  
    - HR dapat menyesuaikan threshold promosi berdasarkan kebutuhan bisnis.
    """)


show_model_analysis()
