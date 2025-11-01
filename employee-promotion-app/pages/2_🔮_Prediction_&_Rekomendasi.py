import streamlit as st
import pandas as pd
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
    if 'Projects' in df.columns and 'YearsAtCompany' in df.columns:
        df['Projects_per_Years'] = df['Projects'] / df['YearsAtCompany']
        df['Projects_per_Years'].replace([np.inf, -np.inf], 0, inplace=True)
        df['Projects_per_Years_log'] = np.log1p(df['Projects_per_Years'])
    return df

def show_prediction_page():
    st.title("🔮 Prediction & Rekomendasi")

    st.markdown("Unggah file data karyawan (.csv) untuk diprediksi apakah layak promosi.")
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file, sep=';')
        st.write("Data yang diunggah:")
        st.dataframe(df.head())
        
        df = feature_engineering(df)
        
        model = load_model()
        pred = model.predict(df)
        df['promotion_prediction'] = pred

        st.success("✅ Prediksi selesai!")
        st.dataframe(df.head())

        st.subheader("📈 Rekomendasi")
        st.markdown("""
        - Tingkatkan **leadership score** dan **training score** untuk karyawan yang belum layak promosi.  
        - Evaluasi ulang performa tahunan dan kontribusi proyek besar.
        """)

show_prediction_page()

