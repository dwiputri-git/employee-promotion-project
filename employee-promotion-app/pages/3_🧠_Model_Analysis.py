import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Model Analysis", layout="wide")

# ------------------ LOAD DATA & MODEL ------------------
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


# ------------------ VISUALIZATION HELPERS ------------------
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)


def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    st.pyplot(fig)
    return roc_auc


# ------------------ MAIN PAGE ------------------
def show_model_analysis():
    st.title("🧠 Model Analysis")
    st.markdown("""
    Halaman ini menampilkan **evaluasi performa model Random Forest**
    yang digunakan untuk memprediksi kelayakan promosi karyawan.
    """)

    df = load_data()
    model, feature_columns = load_model()

    if "Promotion_Eligible" not in df.columns:
        st.error("Kolom 'Promotion_Eligible' tidak ditemukan di dataset.")
        return

    # --- Prediction ---
    X = df[feature_columns]
    y_true = df["Promotion_Eligible"]
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # ------------------ SUMMARY METRICS (TOP CARDS) ------------------
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = auc(*roc_curve(y_true, y_prob)[:2])

    st.markdown("### 📈 Model Performance Summary")
    col1, col2, col3, col4, col5 = st.columns(5)

    card_style = """
        background-color: #f0f6ff;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0px 1px 3px rgba(0,0,0,0.1);
    """

    col1.markdown(f"<div style='{card_style}'><h4>Accuracy</h4><h2>{acc:.3f}</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div style='{card_style}'><h4>Precision</h4><h2>{prec:.3f}</h2></div>", unsafe_allow_html=True)
    col3.markdown(f"<div style='{card_style}'><h4>Recall</h4><h2>{rec:.3f}</h2></div>", unsafe_allow_html=True)
    col4.markdown(f"<div style='{card_style}'><h4>F1-Score</h4><h2>{f1:.3f}</h2></div>", unsafe_allow_html=True)
    col5.markdown(f"<div style='{card_style}'><h4>AUC</h4><h2>{roc_auc:.3f}</h2></div>", unsafe_allow_html=True)

    st.markdown("---")

    # ------------------ LAYOUT: CONFUSION MATRIX | ROC | FEATURE IMPORTANCE ------------------
    col_left, col_mid, col_right = st.columns(3)

    with col_left:
        st.subheader("📉 Confusion Matrix")
        plot_confusion_matrix(y_true, y_pred)

    with col_mid:
        st.subheader("📈 ROC Curve")
        plot_roc_curve(y_true, y_prob)

    with col_right:
        st.subheader("🏗️ Feature Importance")
        try:
            # Ambil feature importances
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

                top_features = feature_importances.head(10)
                fig = px.bar(
                    top_features.sort_values("Importance"),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale="Blues",
                    text=top_features["Importance"].apply(lambda x: f"{x:.3f}")
                )
                fig.update_layout(
                    xaxis_title="Importance",
                    yaxis_title="Feature",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12),
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Model tidak memiliki atribut 'feature_importances_'.")
        except Exception as e:
            st.warning(f"Gagal menampilkan feature importance: {e}")

    # ------------------ CLASSIFICATION REPORT (FULL TABLE) ------------------
    st.markdown("---")
    st.subheader("📊 Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(
        report_df.style.background_gradient(cmap="Blues"),
        use_container_width=True
    )

    # ------------------ FOOTER NOTES ------------------
    st.markdown("""
    ---
    **Catatan Analisis:**
    - Model dengan AUC tinggi berarti lebih baik membedakan antara karyawan yang layak dan tidak layak promosi.  
    - Jika *Precision* rendah, ada risiko merekomendasikan promosi untuk karyawan yang belum siap.  
    - HR dapat menyesuaikan ambang probabilitas promosi sesuai kebijakan organisasi.
    """)


# Jalankan halaman
show_model_analysis()
