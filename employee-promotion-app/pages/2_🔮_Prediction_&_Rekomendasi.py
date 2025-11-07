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
st.write("Pilih cara input data karyawan untuk diprediksi apakah layak promosi.")


# ====================== OPSI INPUT ======================
input_mode = st.radio("Pilih metode input:", ["📂 Upload File CSV"], horizontal=True)


# ====================== LOAD MODEL ======================
model = load_model()


# ====================== INPUT MODE 1 - UPLOAD CSV ======================
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
            df = feature_engineering(df_raw)

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
                df_result["Recommendation"] = np.where(df_result["Probability"] >= 70, "Promote", "Not Ready")

                # ====== TAMPILKAN ======
                st.markdown("### 📋 Hasil Prediksi")
                def highlight_recommendation(val):
                    color = "#b6e7a6" if val == "Promote" else "#f8c8c8"
                    return f"background-color: {color}"

                st.dataframe(
                    df_result.style
                    .applymap(highlight_recommendation, subset=["Recommendation"])
                    .format({"Probability": "{:.2f}%"})
                )

                csv = df_result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="💾 Download Hasil Prediksi",
                    data=csv,
                    file_name="hasil_prediksi_promosi.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")


# ====================== INPUT MODE 2 - MANUAL FORM ======================
elif input_mode == "✍️ Input Manual":
    st.markdown("### 🧾 Isi Data Karyawan")

    with st.form("manual_input_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 18, 65, 30)
            years_at_company = st.number_input("Years at Company", 0, 40, 5)
            training_hours = st.number_input("Training Hours", 0, 500, 40)
        with col2:
            leadership_score = st.number_input("Leadership Score", 0.0, 1.0, 0.6)
            projects_handled = st.number_input("Projects Handled", 0, 100, 5)
            previous_year_rating = st.slider("Previous Year Rating", 0.0, 5.0, 3.5, 0.5)
        with col3:
            department = st.selectbox("Department", ["Sales", "Technical", "Operations", "HR", "Finance", "IT"])
            education = st.selectbox("Education", ["Bachelor", "Master", "PhD"])
            gender = st.selectbox("Gender", ["Male", "Female"])

        submitted = st.form_submit_button("🔍 Prediksi Sekarang")

    if submitted:
        # ====== BENTUKKAN DATAFRAME ======
        data_input = pd.DataFrame({
            "Age": [age],
            "Years_at_Company": [years_at_company],
            "Training_Hours": [training_hours],
            "Leadership_Score": [leadership_score],
            "Projects_Handled": [projects_handled],
            "Previous_Year_Rating": [previous_year_rating],
            "Department": [department],
            "Education": [education],
            "Gender": [gender]
        })

        # ====== FEATURE ENGINEERING ======
        data_input = feature_engineering(data_input)

        if hasattr(model, "feature_names_in_"):
            expected_features = model.feature_names_in_
        else:
            expected_features = data_input.columns

        missing_cols = [col for col in expected_features if col not in data_input.columns]

        if missing_cols:
            st.warning(f"⚠️ Beberapa kolom tidak ditemukan: {', '.join(missing_cols)} — model tetap akan dijalankan jika bisa.")
            for col in missing_cols:
                data_input[col] = 0  # isi default 0 untuk kolom hilang

        # ====== PREDIKSI ======
        preds = model.predict(data_input[expected_features])
        probs = model.predict_proba(data_input[expected_features])[:, 1] * 100
        recommendation = "Promote" if probs[0] >= 70 else "Not Ready"

        # ====== HASIL ======
        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")
        st.metric("Probabilitas Siap Promosi", f"{probs[0]:.2f}%")
        st.metric("Rekomendasi", recommendation, delta=None)

        if recommendation == "Promote":
            st.success("✅ Karyawan ini direkomendasikan untuk promosi.")
        else:
            st.warning("⚠️ Karyawan ini belum siap untuk promosi.")
