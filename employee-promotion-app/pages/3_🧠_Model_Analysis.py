import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Model Analysis", layout="wide")

@st.cache_resource
def load_model():
    base_path = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_path, "model", "model.pkl")
    model = joblib.load(model_path)
    return model

def show_model_analysis():
    st.title("🧠 Model Analysis")

    model = load_model()
    st.markdown("Analisis feature importance dan interpretasi model.")

    # Contoh dummy data
    base_path = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_path, "data", "employee_promotion_dataset.csv")
    df = pd.read_csv(data_path)
    X = df.drop(columns=['Promotion_Eligible'], errors='ignore')

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    st.subheader("📊 Feature Importance (SHAP)")
    fig, ax = plt.subplots()
    shap.plots.bar(shap_values, show=False)
    st.pyplot(fig)

show_model_analysis()

