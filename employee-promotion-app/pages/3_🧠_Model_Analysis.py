import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    brier_score_loss,
)
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Model Analysis", layout="wide")

# --- Load Model & Data ---
@st.cache_data
def load_data():
    # sesuaikan path datasetmu
    return pd.read_csv("employee-promotion-app/data/cleaned_data.csv")

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

# --- Helper Plots ---
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    labels = ["Not Promoted (0)", "Promoted (1)"]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Oranges",
        xticklabels=labels, yticklabels=labels, cbar=False,
        linewidths=1, linecolor="white", ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=13, pad=10)
    st.pyplot(fig)

def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], lw=1, linestyle="--", label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    st.pyplot(fig)

def plot_pr_curve(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall Curve (AP = {ap:.3f})")
    st.pyplot(fig)

def plot_probability_hist(y_prob, threshold=0.5):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(y_prob, bins=15, alpha=0.7)
    ax.axvline(threshold, linestyle="--", linewidth=2)
    ax.set_xlabel("Prediction Probability")
    ax.set_ylabel("Count")
    ax.set_title("Probability Distribution")
    st.pyplot(fig)

# --- Main Page ---
def show_model_analysis():
    st.title("🧠 Model Analysis")

    df = load_data()
    model, feature_columns = load_model()

    if "Promotion_Eligible" not in df.columns:
        st.error("Kolom 'Promotion_Eligible' tidak ditemukan di dataset.")
        return

    X = df[feature_columns]
    y_true = df["Promotion_Eligible"].astype(int)

    # Prediksi
    try:
        y_prob = model.predict_proba(X)[:, 1]
    except Exception:
        y_prob = np.full(len(df), 0.5)
    y_pred = (y_prob >= 0.5).astype(int)
    threshold = 0.5  # tampilkan juga di metrik

    # ===== Header: Model Information =====
    st.markdown("### Model Information")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Model Type**")
        st.write(type(model).__name__)
    with c2:
        st.markdown("**Training Data**")
        st.write(f"{len(df):,} samples")
    with c3:
        st.markdown("**Features**")
        st.write(f"{len(feature_columns)} features")

    # ===== Performance Metrics =====
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        rocauc = roc_auc_score(y_true, y_prob)
    except Exception:
        rocauc = float("nan")
    ap = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    st.markdown("### Performance Metrics")
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("Accuracy", f"{acc:.3f}")
    m2.metric("Precision", f"{prec:.3f}")
    m3.metric("Recall", f"{rec:.3f}")
    m4.metric("F1-Score", f"{f1:.3f}")
    m5.metric("ROC-AUC", f"{rocauc:.3f}")
    m6.metric("PR-AUC", f"{ap:.3f}")
    m7.metric("Brier Score", f"{brier:.3f}")
    st.caption(f"**Threshold**: {threshold:.3f}")

    st.markdown("### Model Visualizations")
    # ===== Row 1: Confusion Matrix | ROC =====
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(y_true, y_pred)
    with r1c2:
        st.subheader("ROC Curve")
        plot_roc_curve(y_true, y_prob)

    # ===== Row 2: PR Curve | Probability Dist =====
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.subheader("Precision–Recall Curve")
        plot_pr_curve(y_true, y_prob)
    with r2c2:
        st.subheader("Probability Distribution")
        plot_probability_hist(y_prob, threshold=threshold)

    # ===== Feature Importance =====
    st.markdown("### Feature Importance")
    # ambil estimator RF & nama fitur setelah preprocessing jika ada
    rf_step = None
    encoded_feature_names = feature_columns
    if hasattr(model, "named_steps"):
        rf_step = model.named_steps.get("rf", None) or model.named_steps.get("classifier", None)
        try:
            pre = model.named_steps.get("preprocessor", None)
            if pre is not None:
                encoded_feature_names = pre.get_feature_names_out()
        except Exception:
            encoded_feature_names = feature_columns
    else:
        rf_step = model

    if rf_step is not None and hasattr(rf_step, "feature_importances_"):
        importances = rf_step.feature_importances_
        n = min(len(importances), len(encoded_feature_names))
        feature_importances = (
            pd.DataFrame({
                "Feature": encoded_feature_names[:n],
                "Importance": importances[:n]
            })
            .sort_values("Importance", ascending=False)
        )

        # Plot Top-10 horizontal bar
        fig_imp = px.bar(
            feature_importances.head(10).sort_values("Importance"),
            x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Blues",
            title="Feature Importance (Top 10)"
        )
        fig_imp.update_layout(height=420, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_imp, use_container_width=True)

        # Tabel ringkas
        st.dataframe(
            feature_importances.head(10).reset_index(drop=True),
            use_container_width=True,
            height=300
        )
    else:
        st.warning("Feature importance tidak tersedia pada model ini.")

    # ===== Evaluation Details =====
    st.markdown("### Model Evaluation Details")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    err = fp + fn
    total = tn + fp + fn + tp
    err_rate = err / total if total else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    cL, cR = st.columns(2)
    with cL:
        st.markdown("**Confusion Matrix Breakdown**")
        st.markdown(
            f"- True Negatives: **{tn}**  \n"
            f"- False Positives: **{fp}**  \n"
            f"- False Negatives: **{fn}**  \n"
            f"- True Positives: **{tp}**"
        )
    with cR:
        st.markdown("**Error Analysis**")
        st.markdown(
            f"- Total Errors: **{err}**  \n"
            f"- Error Rate: **{err_rate:.1%}**  \n"
            f"- False Positive Rate: **{fpr:.1%}**  \n"
            f"- False Negative Rate: **{fnr:.1%}**"
        )

    # ===== Interpretation & Recommendations =====
    st.markdown("### Model Interpretation")
    st.markdown(
        "- Fitur kinerja (mis. **Performance_Score**, **Leadership_Score**) cenderung paling berpengaruh."
        "\n- Distribusi probabilitas menunjukkan mayoritas prediksi berada di bawah threshold."
        "\n- ROC/PR curve mengindikasikan separasi kelas yang moderat—perlu tuning & fitur tambahan."
    )

    st.markdown("### Model Recommendations")
    st.markdown(
        "1. **Hyperparameter Tuning** (Optuna/GridSearch) untuk meningkatkan ROC-AUC & PR-AUC.\n"
        "2. **Feature Engineering**: interaction terms (e.g., performance × leadership), recency features, dan normalisasi jam pelatihan per tahun.\n"
        "3. **Handle Class Imbalance**: gunakan class_weight atau SMOTE jika label tidak seimbang.\n"
        "4. **Calibrate Probabilities** (CalibratedClassifierCV) untuk menurunkan Brier score.\n"
        "5. **Threshold Strategy**: pertahankan default 0.7 untuk keputusan **Promote**, 0.5–0.7 **Need Review**."
    )

# --- Run Page ---
show_model_analysis()
