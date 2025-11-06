import streamlit as st

st.set_page_config(
    page_title="Employee Promotion App",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 Employee Promotion Prediction App")
st.markdown("""
Selamat datang di **Employee Promotion App**!  
Gunakan sidebar di kiri untuk menavigasi ke:
- 📊 Dashboard
- 🔮 Prediction & Rekomendasi
- 🧠 Model Analysis
""")

import pickle
import pandas as pd

# Load model dan fitur
with open("models/rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Contoh: data input user
sample_data = pd.DataFrame([{
    "Age": 35,
    "Years_at_Company": 7,
    "Performance_Score": 85,
    "Leadership_Score": 78,
    "Training_Hours": 40,
    "Projects_Handled": 6,
    "Peer_Riview_Score": 82,
    "Current_Position_Level": 3,
}])

# Pastikan kolomnya sesuai
X = sample_data.reindex(columns=feature_columns, fill_value=0)

# Prediksi
prediction = model.predict(X)
probability = model.predict_proba(X)[:, 1]

st.write(f"**Prediction:** {'Promote' if prediction[0] == 1 else 'Not Ready'}")
st.write(f"**Probability:** {probability[0]:.2f}")
