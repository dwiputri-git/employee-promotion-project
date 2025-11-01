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
    st.markdown("Analisis feature importance dan interpretasi model menggunakan SHAP.")

    model = load_model()

    # --- Load sample data
    base_path = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_path, "data", "employee_promotion_dataset.csv")
    df = pd.read_csv(data_path, sep=';')
    X = df.drop(columns=['Promotion_Eligible'], errors='ignore')

    # --- Pisahkan preprocessor dan model RandomForest dari pipeline
    preprocessor = model.named_steps['preprocessor']
    rf_model = model.named_steps['rf']

    # --- Transform data sesuai preprocessing pipeline
    X_transformed = preprocessor.transform(X)

    # --- Ambil nama fitur hasil transformasi
    ohe = preprocessor.named_transformers_['cat']
    cat_features = ohe.get_feature_names_out(['Current_Position_Level'])
    num_features = ['Age', 'Years_at_Company', 'Performance_Score', 'Leadership_Score',
                    'Training_Hours', 'Projects_Handled', 'Peer_Review_Score', 'Projects_per_Years_log']
    feature_names = list(num_features) + list(cat_features)

    # --- Jalankan SHAP untuk model RandomForest
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_transformed)

    # --- Tampilkan hasil di Streamlit
    st.subheader("📊 Feature Importance (SHAP Summary Plot)")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values[1], X_transformed, feature_names=feature_names, show=False)
    st.pyplot(fig)

    with st.expander("📈 Detail Feature Impact"):
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values[1], X_transformed, feature_names=feature_names, plot_type="bar", show=False)
        st.pyplot(fig2)

    st.markdown("""
    **Interpretasi singkat:**
    - Fitur di bagian atas memiliki pengaruh paling besar terhadap keputusan promosi.
    - Warna biru → nilai rendah, warna merah → nilai tinggi pada fitur tersebut.
    """)

show_model_analysis()
