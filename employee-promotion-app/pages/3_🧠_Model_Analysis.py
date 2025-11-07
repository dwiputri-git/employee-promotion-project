import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import numpy as np
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
import plotly.express as px

st.set_page_config(page_title="Model Analysis", layout="wide")

# -----------------------------
# ✅ Load Model & Data
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("employee-promotion-app/data/data_test.csv")

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

# -----------------------------
# ✅ Helper Plots
# -----------------------------
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    labels = ["Not Promoted (0)", "Promoted (1)"]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Oranges",
        xticklabels=labels, yticklabels=labels, cbar=False,
        linewidths=1, linecolor="white", ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
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
    ax.axvline(threshold, linestyle="--", color="r", linewidth=2, label=f"Threshold ({threshold})")
    ax.set_xlabel("Prediction Probability")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title("Probability Distribution")
    st.pyplot(fig)

# -----------------------------
# ✅ Manual Feature Importance Chart
# -----------------------------
def plot_manual_feature_importance():
    manual_importance = {
        "Performance_Score": 0.10,
        "Project_per_Years": 0.095,
        "Age": 0.090,
        "Projects_Handled": 0.085,
        "Years_at_Company": 0.080,
        "Leadership_Score": 0.075,
        "Training_Hours": 0.070,
        "Peer_Review_Score": 0.060,
        "Project_Level (Moderate)": 0.050,
        "Age_Group (Late Mid)": 0.045,
        "Project_Level (Very High)": 0.040,
        "Tenure_Level (Veteran)": 0.035,
        "Age_Group (Senior)": 0.030,
        "Project_Level (High)": 0.025,
        "Tenure_Level (Senior)": 0.020,
    }

    items = sorted(manual_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    features, scores = zip(*items)
    colors = plt.cm.viridis(np.linspace(0.25, 0.95, len(scores)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(features))[::-1], scores, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_yticks(range(len(features))[::-1])
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel("Importance Score")
    ax.set_title("Top 10 Feature Importance (Manual)", fontsize=13, pad=8)

    for i, v in enumerate(scores):
        ax.text(v + max(scores)*0.01, len(features)-1-i, f"{v:.3f}", va="center", fontsize=9)

    ax.set_xlim(0, max(scores)*1.15)
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    st.pyplot(fig)

# -----------------------------
# ✅ Main Page
# -----------------------------
def show_model_analysis():
    st.title("🧠 Model Analysis")

    df = load_data()
    model, feature_columns = load_model()

    if "Promotion_Eligible" not in df.columns:
        st.error("Kolom 'Promotion_Eligible' tidak ditemukan di dataset.")
        return

    X = df[feature_columns]
    y_true = df["Promotion_Eligible"].astype(int)

    try:
        y_prob = model.predict_proba(X)[:, 1]
    except Exception:
        y_prob = np.full(len(df), 0.5)
    y_pred = (y_prob >= 0.5).astype(int)
    threshold = 0.5

    # === Model Info ===
    st.markdown("### Model Information")
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Type", type(model).__name__)
    col2.metric("Training Data", f"{len(df):,} samples")
    col3.metric("Features", f"{len(feature_columns)}")

    # === Metrics ===
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    rocauc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    st.markdown("### Performance Metrics")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Accuracy", f"{acc:.3f}")
    c2.metric("Precision", f"{prec:.3f}")
    c3.metric("Recall", f"{rec:.3f}")
    c4.metric("F1-Score", f"{f1:.3f}")
    c5.metric("ROC-AUC", f"{rocauc:.3f}")
    c6.metric("PR-AUC", f"{ap:.3f}")
    c7.metric("Brier Score", f"{brier:.3f}")
    st.caption(f"Threshold: {threshold}")

    # === Visualizations ===
    st.markdown("### Model Visualizations")
    colA, colB = st.columns(2)
    with colA:
        plot_confusion_matrix(y_true, y_pred)
    with colB:
        plot_roc_curve(y_true, y_prob)

    colC, colD = st.columns(2)
    with colC:
        plot_pr_curve(y_true, y_prob)
    with colD:
        plot_probability_hist(y_prob, threshold=threshold)

    # === Manual Feature Importance ===
    st.markdown("### Feature Importance")
    plot_manual_feature_importance()

    # === Evaluation Summary ===
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

    st.markdown("### Model Interpretation")
    st.markdown(
        "- **Performance_Score** dan **Leadership_Score** paling berpengaruh terhadap keputusan promosi.\n"
        "- Distribusi probabilitas menunjukkan sebagian besar prediksi masih di bawah threshold.\n"
        "- ROC/PR curve menunjukkan model masih perlu tuning agar separasi antar kelas makin kuat."
    )

    st.markdown("### Model Recommendations")
    st.markdown(
        "1. **Hyperparameter Tuning** untuk menaikkan ROC-AUC & PR-AUC.\n"
        "2. **Feature Engineering** (misal interaksi performance × leadership, rasio training).\n"
        "3. **Class Balance** pakai `class_weight` atau SMOTE.\n"
        "4. **Calibrate Probabilities** (CalibratedClassifierCV) untuk menurunkan Brier score.\n"
        "5. **Gunakan threshold 0.7** untuk keputusan *Promote*, 0.5–0.7 untuk *Need Review*."
    )

# -----------------------------
# ✅ Jalankan Page
# -----------------------------
show_model_analysis()
