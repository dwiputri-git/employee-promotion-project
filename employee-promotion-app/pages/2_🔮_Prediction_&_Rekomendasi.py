import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Prediction & Rekomendasi", layout="wide")

# ====== LOAD MODEL ======
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "..", "rf_model2.pkl")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


# ====== FUNGSI PREPROCESSING ======
def preprocess_data(df):
    df = df.copy()

    # --- Contoh encoding / feature engineering sederhana ---
    # Pastikan sesuaikan dengan yang kamu pakai saat training
    if "Training_Score" in df.columns:
        df["Training_Level"] = pd.cut(
            df["Training_Score"],
            bins=[0, 40, 60, 80, 100],
            labels=["Low", "Medium", "High", "Excellent"]
        )

    if "Leadership_Score" in df.columns:
        df["Leadership_Level"] = pd.cut(
            df["Leadership_Score"],
            bins=[0, 40, 60, 80, 100],
            labels=["Low", "Medium", "High", "Excellent"]
        )

    if "Projects_per_Years" in df.columns:
        df["Projects_per_Years_log"] = np.log1p(df["Projects_per_Years"])

    if "Tenure" in df.columns:
        df["Tenure_Level"] = pd.cut(
            df["Tenure"],
            bins=[0, 2, 5, 10, 20, 40],
            labels=["New", "Junior", "Mid", "Senior", "Expert"]
        )

    if "Age" in df.columns:
        df["Age_Group"] = pd.cut(
            df["Age"],
            bins=[18, 25, 35, 45, 55, 65],
            labels=["Young", "Early Mid", "Mid", "Senior", "Late Senior"]
        )

    # --- Project Level dari nilai performance misalnya ---
    if "Project_Score" in df.columns:
        df["Project_Level"] = pd.cut(
            df["Project_Score"],
            bins=[0, 40, 60, 80, 100],
            labels=["Low", "Medium", "High", "Excellent"]
        )

    return df


# ====== STYLE ======
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


# ====== PAGE TITLE ======
st.title("🔮 Prediction & Rekomendasi")
st.write("Unggah file data karyawan (.csv) untuk diprediksi apakah layak promosi.")


# ====== FILE UPLOAD ======
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
        st.markdown("---")

        # Preprocess data agar sesuai model
        df = preprocess_data(df_raw)

        # Load model
        model = load_model()
        expected_features = model.feature_names_in_

        # Pastikan semua fitur tersedia
        missing_cols = [col for col in expected_features if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Kolom berikut masih belum ada walau sudah diproses: {', '.join(missing_cols)}")
        else:
            X_new = df[expected_features]

            # Prediksi dan probabilitas
            preds = model.predict(X_new)
            probs = model.predict_proba(X_new)[:, 1] * 100

            df_raw["Prediction"] = preds
            df_raw["Probability"] = probs
            df_raw["Recommendation"] = np.where(df_raw["Probability"] >= 70, "Promote", "Not Ready")

            # Tampilkan hasil
            st.markdown("### 📋 Hasil Prediksi")
            st.write("Berikut hasil prediksi kelayakan promosi:")

            def highlight_recommendation(val):
                color = "#b6e7a6" if val == "Promote" else "#f8c8c8"
                return f"background-color: {color}"

            st.dataframe(
                df_raw.style
                .applymap(highlight_recommendation, subset=["Recommendation"])
                .format({"Probability": "{:.2f}%"})
            )

            # Tombol download
            csv = df_raw.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="💾 Download Hasil Prediksi",
                data=csv,
                file_name="hasil_prediksi_promosi.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")
