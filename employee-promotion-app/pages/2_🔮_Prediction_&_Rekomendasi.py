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


# ====================== PAGE TITLE ======================
st.title("🔮 Prediction & Rekomendasi Promosi")
st.markdown("Prediksi kelayakan promosi berdasarkan kinerja dan karakteristik karyawan.")


# ====================== INPUT MANUAL ======================
st.markdown("### ✍️ Input Data Karyawan")

with st.form("manual_input_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        employee_id = st.text_input("Employee ID", "EMP9999")
        age = st.number_input("Age", 20, 60, 30)
        years_at_company = st.number_input("Years at Company", 0, 30, 5)
    with col2:
        performance_score = st.slider("Performance Score", 1, 5, 3)
        leadership_score = st.slider("Leadership Score (1–100)", 1.0, 100.0, 75.0)
        training_hours = st.number_input("Training Hours (1–200)", 1, 200, 50)
    with col3:
        projects_handled = st.number_input("Projects Handled", 0, 20, 6)
        peer_review_score = st.slider("Peer Review Score (1–100)", 1.0, 100.0, 80.0)
        current_position_level = st.selectbox("Current Position Level", ['Junior','Mid','Senior','Lead'])

    submitted = st.form_submit_button("🔍 Prediksi Sekarang")


# ====================== PREDIKSI ======================
if submitted:
    model = load_model()

    # ====== DataFrame Input ======
    data_input = pd.DataFrame({
        "Employee_ID": [employee_id],
        "Age": [age],
        "Years_at_Company": [years_at_company],
        "Performance_Score": [performance_score],
        "Leadership_Score": [leadership_score],
        "Training_Hours": [training_hours],
        "Projects_Handled": [projects_handled],
        "Peer_Review_Score": [peer_review_score],
        "Current_Position_Level": [current_position_level]
    })

    # ====== Feature Engineering ======
    data_input = feature_engineering(data_input)

    # ====== Encode kategorikal manual (karena ada string) ======
    data_encoded = pd.get_dummies(data_input, drop_first=True)

    # ====== Cocokkan dengan model feature columns ======
    if hasattr(model, "feature_names_in_"):
        expected_features = model.feature_names_in_
    else:
        expected_features = data_encoded.columns

    for col in expected_features:
        if col not in data_encoded.columns:
            data_encoded[col] = 0  # tambahkan kolom yang hilang

    X_input = data_encoded[expected_features]

    # ====== Prediksi ======
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][1] * 100
    recommendation = "Promote" if prob >= 70 else "Not Ready"

    # ====== Hasil ======
    st.markdown("---")
    st.subheader("📊 Hasil Prediksi")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Probabilitas Siap Promosi", f"{prob:.2f}%")
    with col_b:
        st.metric("Rekomendasi", recommendation)

    if recommendation == "Promote":
        st.success("✅ Karyawan ini direkomendasikan untuk promosi.")
    else:
        st.warning("⚠️ Karyawan ini belum siap untuk promosi.")

    # ====== Tabel Ringkas ======
    st.markdown("### 📋 Detail Input")
    st.dataframe(data_input.T, use_container_width=True)
