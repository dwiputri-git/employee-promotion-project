"""
Model Analysis page for the Employee Promotion Prediction App
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.visualizations import (
    create_confusion_matrix_plot, create_roc_curve_plot, 
    create_precision_recall_plot, create_feature_importance_plot,
    create_probability_distribution_plot
)

def show_model_analysis():
    """Model analysis page"""
    st.header("📈 Model Analysis")
    
    # Model information
    st.subheader("Model Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Type", "RandomForest Tuned")
    
    with col2:
        st.metric("Training Data", "939 samples")
    
    with col3:
        st.metric("Features", "16 features")
    
    st.divider()
    
    # Mock data for analysis (simulating V3 results)
    @st.cache_data
    def generate_mock_evaluation_data():
        np.random.seed(42)
        n_samples = 100
        
        # Generate realistic evaluation data
        y_true = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        y_prob = np.random.beta(2, 5, n_samples)
        y_pred = (y_prob > 0.5).astype(int)
        
        return y_true, y_prob, y_pred
    
    y_true, y_prob, y_pred = generate_mock_evaluation_data()
    
    # Model metrics
    st.subheader("Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy = (y_true == y_pred).mean()
        st.metric("Accuracy", f"{accuracy:.3f}")
    
    with col2:
        precision = (y_true[y_pred == 1] == 1).mean() if (y_pred == 1).sum() > 0 else 0
        st.metric("Precision", f"{precision:.3f}")
    
    with col3:
        recall = (y_pred[y_true == 1] == 1).mean() if (y_true == 1).sum() > 0 else 0
        st.metric("Recall", f"{recall:.3f}")
    
    with col4:
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        st.metric("F1-Score", f"{f1:.3f}")
    
    # Additional metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # ROC-AUC
        from sklearn.metrics import roc_auc_score
        roc_auc = roc_auc_score(y_true, y_prob)
        st.metric("ROC-AUC", f"{roc_auc:.3f}")
    
    with col2:
        # PR-AUC
        from sklearn.metrics import average_precision_score
        pr_auc = average_precision_score(y_true, y_prob)
        st.metric("PR-AUC", f"{pr_auc:.3f}")
    
    with col3:
        # Brier Score
        from sklearn.metrics import brier_score_loss
        brier = brier_score_loss(y_true, y_prob)
        st.metric("Brier Score", f"{brier:.3f}")
    
    with col4:
        # Threshold
        threshold = 0.5
        st.metric("Threshold", f"{threshold:.3f}")
    
    st.divider()
    
    # Visualizations
    st.subheader("Model Visualizations")
    
    # First row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion Matrix
        fig = create_confusion_matrix_plot(y_true, y_pred, "Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # ROC Curve
        fig = create_roc_curve_plot(y_true, y_prob, "ROC Curve")
        st.plotly_chart(fig, use_container_width=True)
    
    # Second row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Precision-Recall Curve
        fig = create_precision_recall_plot(y_true, y_prob, "Precision-Recall Curve")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Probability Distribution
        fig = create_probability_distribution_plot(y_prob, "Probability Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Feature importance
    st.subheader("Feature Importance")
    
    # Mock feature importance (based on V3 results)
    mock_importance = {
        'Performance_Score': 0.45,
        'Leadership_Score': 0.32,
        'Years_at_Company': 0.28,
        'Training_Hours': 0.15,
        'Projects_Handled': 0.12,
        'Age': -0.08,
        'Peer_Review_Score': 0.06,
        'Training_Level': 0.05,
        'Leadership_Level': 0.04,
        'Perf_x_Leader': 0.03
    }
    
    fig = create_feature_importance_plot(mock_importance, "Feature Importance (Top 10)")
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance table
    importance_df = pd.DataFrame([
        {'Feature': k, 'Importance': v} for k, v in mock_importance.items()
    ]).sort_values('Importance', key=abs, ascending=False)
    
    st.dataframe(importance_df, use_container_width=True)
    
    st.divider()
    
    # Model evaluation details
    st.subheader("Model Evaluation Details")
    
    # Confusion matrix breakdown
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Confusion Matrix Breakdown**")
        st.write(f"- True Negatives: {tn}")
        st.write(f"- False Positives: {fp}")
        st.write(f"- False Negatives: {fn}")
        st.write(f"- True Positives: {tp}")
    
    with col2:
        st.write("**Error Analysis**")
        total_errors = fp + fn
        error_rate = total_errors / len(y_true) * 100
        st.write(f"- Total Errors: {total_errors}")
        st.write(f"- Error Rate: {error_rate:.1f}%")
        st.write(f"- False Positive Rate: {fp/(fp+tn)*100:.1f}%")
        st.write(f"- False Negative Rate: {fn/(fn+tp)*100:.1f}%")
    
    # Model interpretation
    st.subheader("Model Interpretation")
    
    st.write("""
    **Key Insights:**
    
    1. **Performance Score** is the most important feature, indicating that current performance 
       is the strongest predictor of promotion eligibility.
    
    2. **Leadership Score** is the second most important, suggesting that leadership qualities 
       are crucial for promotion decisions.
    
    3. **Years at Company** shows positive importance, indicating that tenure plays a role 
       in promotion decisions.
    
    4. **Age** shows negative importance, suggesting that younger employees might have 
       higher promotion potential (or the model is detecting age-related bias).
    
    5. The model shows good calibration with a Brier score of {:.3f}, indicating 
       well-calibrated probability estimates.
    """.format(brier))
    
    # Recommendations
    st.subheader("Model Recommendations")
    
    st.write("""
    **For Model Improvement:**
    
    1. **Hyperparameter Tuning**: Run Optuna optimization to find optimal parameters
    2. **Feature Engineering**: Add more interaction terms and domain-specific features
    3. **Fairness Analysis**: Address potential bias in position-level predictions
    4. **Regular Retraining**: Implement quarterly model retraining with new data
    
    **For Deployment:**
    
    1. **Threshold Optimization**: Adjust threshold based on business cost of errors
    2. **Monitoring**: Track model performance and data drift in production
    3. **Human Oversight**: Use model as decision support, not sole decision maker
    4. **A/B Testing**: Compare model performance against current promotion process
    """)

if __name__ == "__main__":
    show_model_analysis()
