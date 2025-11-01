import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Prediction", layout="wide")

@st.cache_resource
def load_model():
    base_path = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_path, "model", "model.pkl")
    model = joblib.load(model_path)
    return model

def feature_engineering(df):
    """Lakukan feature engineering yang sama seperti di training."""
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

def show_prediction_page():
    st.title("🔮 Prediction & Rekomendasi")

    st.markdown("Unggah file data karyawan (.csv) untuk diprediksi apakah layak promosi.")
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file, sep=';')
        st.write("Data yang diunggah:")
        st.dataframe(df, height=400, use_container_width=False)

        model = load_model()

        # 🧩 Feature engineering
        df = feature_engineering(df)
        df_display = df.copy()  # simpan versi lengkap untuk ditampilkan

        # 🧩 Samakan kolom dengan model
        expected_cols = model.feature_names_in_
        missing_cols = [col for col in expected_cols if col not in df.columns]

        if missing_cols:
            st.warning(f"Menambahkan kolom yang hilang: {missing_cols}")
            for col in missing_cols:
                df[col] = 0

        df_model = df[expected_cols]  # versi buat model

        # 🔮 Prediksi
        pred = model.predict(df_model)
        df_display['promotion_prediction'] = pred  # tambahkan hasil prediksi ke versi tampilan

        # 🚫 Hilangkan kolom log agar tampilan lebih bersih
        if 'Projects_per_Years_log' in df_display.columns:
            df_display = df_display.drop(columns=['Projects_per_Years_log'])

        st.success("✅ Prediksi selesai!")
        st.dataframe(df_display, height=400, use_container_width=False)
        
        st.subheader("📈 Rekomendasi")
        st.markdown("""
        - Tingkatkan **leadership score** dan **training score** untuk karyawan yang belum layak promosi.  
        - Evaluasi ulang performa tahunan dan kontribusi proyek besar.
        """)

show_prediction_page()
