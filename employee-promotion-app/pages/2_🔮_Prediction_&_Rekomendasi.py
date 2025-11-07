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


# ====== STYLE ======
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
    }
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
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ File berhasil diunggah: **{uploaded_file.name}**")
        st.markdown("---")

        # ====== LOAD MODEL ======
        model = load_model()

        # ====== CEK KOMPATIBILITAS FITUR ======
        expected_features = model.feature_names_in_
        missing_cols = [col for col in expected_features if col not in df.columns]

        if missing_cols:
            st.error(f"❌ Kolom berikut hilang di file kamu: {', '.join(missing_cols)}")
        else:
            X_new = df[expected_features]

            # ====== PREDIKSI ======
            preds = model.predict(X_new)
            probs = model.predict_proba(X_new)[:, 1] * 100

            df["Prediction"] = preds
            df["Probability"] = probs
            df["Recommendation"] = np.where(df["Probability"] >= 70, "Promote", "Not Ready")

            # ====== HASIL ======
            st.markdown("### 📋 Hasil Prediksi")
            st.write("Berikut hasil prediksi kelayakan promosi untuk data yang kamu unggah:")

            def highlight_recommendation(val):
                color = "#b6e7a6" if val == "Promote" else "#f8c8c8"
                return f"background-color: {color}"

            st.dataframe(
                df.style
                .applymap(highlight_recommendation, subset=["Recommendation"])
                .format({"Probability": "{:.2f}%"})
            )

            # ====== DOWNLOAD HASIL ======
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="💾 Download Hasil Prediksi",
                data=csv,
                file_name="hasil_prediksi_promosi.csv",
                mime="text/csv"
            )

            # ====== CATATAN ======
            st.markdown("""
            <p style='color:gray; font-size:13px;'>
            Kolom <b>Probability</b> menunjukkan keyakinan model terhadap kelayakan promosi.  
            Nilai ≥ 70% → direkomendasikan <b>Promote</b>.
            </p>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")
