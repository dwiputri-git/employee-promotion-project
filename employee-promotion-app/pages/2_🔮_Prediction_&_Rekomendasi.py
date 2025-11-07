import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Prediction & Rekomendasi", layout="wide")

# ====================== LOAD MODEL ======================
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "..", "rf_model2.pkl")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


# ====================== FEATURE ENGINEERING ======================
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


# ====================== STYLE ======================
st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    .upload-box {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 40px;
        text-align: center;
        background-color: #f9f9f9;
    }
    .highlight {
        color: #4B0082;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


# ====================== PAGE TITLE ======================
st.title("🔮 Prediction & Rekomendasi")
st.write("Upload data karyawan untuk diprediksi apakah layak promosi.")


# ====================== OPSI INPUT ======================
input_mode = st.radio("Pilih metode input:", ["📂 Upload File CSV"], horizontal=True)


# ====================== LOAD MODEL ======================
model = load_model()


# ====================== INPUT MODE - UPLOAD CSV ======================
if input_mode == "📂 Upload File CSV":
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"], label_visibility="collapsed")

    if uploaded_file is None:
        st.markdown("""
            <div class="upload-box">
                <p>📂 <b>Drag and drop file di sini</b> atau klik <span class="highlight">Browse files</span></p>
                <p style="color:gray; font-size:13px;">Limit 200MB per file • CSV</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.success(f"✅ File berhasil diunggah: **{uploaded_file.name}**")

            # Pastikan semua kolom numerik jadi integer
            for col in df_raw.select_dtypes(include=['float64', 'int64']).columns:
                df_raw[col] = df_raw[col].astype(int)

            # Feature engineering
            df = feature_engineering(df_raw.copy())

            # Ambil fitur yang sesuai model
            if hasattr(model, "feature_names_in_"):
                expected_features = model.feature_names_in_
            else:
                expected_features = df.columns

            missing_cols = [col for col in expected_features if col not in df.columns]

            if missing_cols:
                st.error(f"❌ Kolom berikut hilang di file kamu: {', '.join(missing_cols)}")
            else:
                # ====== PREDIKSI ======
                X_new = df[expected_features]
                preds = model.predict(X_new)
                probs = model.predict_proba(X_new)[:, 1] * 100

                df_result = df_raw.copy()
                df_result["Prediction"] = preds
                df_result["Probability"] = probs
                df_result["Recommendation"] = np.where(df_result["Probability"] >= 50, "Promote", "Not Ready")

                # Hapus kolom Projects_per_Years_log dari tampilan (jika ada)
                display_cols = [col for col in df_result.columns if col != "Projects_per_Years_log"]

                # ====== TAMPILKAN ======
                st.markdown("### 📋 Hasil Prediksi")
                def highlight_recommendation(val):
                    color = "#b6e7a6" if val == "Promote" else "#f8c8c8"
                    return f"background-color: {color}"

                st.dataframe(
                    df_result[display_cols].style
                    .applymap(highlight_recommendation, subset=["Recommendation"])
                    .format({"Probability": "{:.2f}%"})
                )

                csv = df_result[display_cols].to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="💾 Download Hasil Prediksi",
                    data=csv,
                    file_name="hasil_prediksi_promosi.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")
