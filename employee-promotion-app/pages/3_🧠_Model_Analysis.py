import streamlit as st
import pandas as pd
import numpy as np
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

# 🧩 Sama seperti di training
def feature_engineering(df):
    if 'Projects_Handled' in df.columns and 'Years_at_Company' in df.columns:
        df['Projects_per_Years'] = df['Projects_Handled'] / df['Years_at_Company']
        df['Projects_per_Years'].replace([np.inf, -np.inf], 0, inplace=True)
        df['Projects_per_Years_log'] = np.log1p(df['Projects_per_Years'])
        df['Project_Level'] = pd.qcut(df['Projects_per_Years'], q=4, labels=['Low','Moderate','High','Very High'])
    if 'Training_Hours' in df.columns:
        df['Training_Level'] = pd.qcut(df['Training_Hours'], q=5, labels=['Very Low','Low','Moderate','High','Very High'])
    if 'Leadership_Score' in df.columns:
        df['Leadership_Level'] = pd.qcut(df['Leadership_Score'], q=4, labels=['Low','Medium','High','Very High'])
    if 'Years_at_Company' in df.columns:
        df['Tenure_Level'] = pd.qcut(df['Years_at_Company'], q=4, labels=['New','Mid','Senior','Veteran'])
    if 'Age' in df.columns:
        df['Age_Group'] = pd.qcut(df['Age'], q=4, labels=['Young','Early Mid','Late Mid','Senior'])
    return df

def show_model_analysis():
    st.title("🧠 Model Analysis")
    st.markdown("Analisis feature importance dan interpretasi model menggunakan SHAP.")

    model = load_model()

    # --- Load sample data
    base_path = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_path, "data", "employee_promotion_dataset.csv")
    df = pd.read_csv(data_path, sep=';')
    df = feature_engineering(df)

    X = df.drop(columns=['Promotion_Eligible'], errors='ignore')

    # --- Ambil preprocessor dan model
    preprocessor = model.named_steps['preprocessor']
    rf_model = model.named_steps['rf']

    # --- Transform data
    X_transformed = preprocessor.transform(X)

    # --- Dapatkan nama kolom sebenarnya dari preprocessor
    num_features = preprocessor.transformers_[0][2]
    ohe = preprocessor.named_transformers_['cat']
    cat_features = ohe.get_feature_names_out(preprocessor.transformers_[1][2])
    feature_names = np.concatenate([num_features, cat_features])

    # --- Jalankan SHAP pada model RF
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_transformed)

    # --- Pilih SHAP value yang sesuai (untuk binary classification)
    if isinstance(shap_values, list):
        shap_values_to_use = shap_values[-1]  # kelas positif
    else:
        shap_values_to_use = shap_values

    # --- Pastikan ukuran cocok
    if shap_values_to_use.shape[1] != X_transformed.shape[1]:
        st.warning("⚠️ Ukuran SHAP dan data tidak cocok, mencoba reshape otomatis...")
        shap_values_to_use = shap_values_to_use.reshape(X_transformed.shape[0], -1)

    st.write(f"✅ Data shape: {X_transformed.shape}, SHAP shape: {shap_values_to_use.shape}")

    # --- Pastikan SHAP value dan data dalam DataFrame agar plot bisa tampil
    shap_df = pd.DataFrame(shap_values_to_use, columns=feature_names)
    X_df = pd.DataFrame(X_transformed, columns=feature_names)

    st.subheader("📊 SHAP Summary Plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_df.values, X_df, feature_names=feature_names, show=False)
    st.pyplot(fig, clear_figure=True)

    # --- Plot bar
    with st.expander("📈 SHAP Feature Importance (Bar Chart)"):
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_df.values, X_df, feature_names=feature_names, plot_type="bar", show=False)
        st.pyplot(fig2, clear_figure=True)
    st.markdown("""
    **Interpretasi singkat:**
    - Fitur di bagian atas memiliki pengaruh paling besar terhadap keputusan promosi.
    - Warna biru → nilai rendah, warna merah → nilai tinggi pada fitur tersebut.
    """)

show_model_analysis()
