import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =====================================================
# 🎯 CONFIG
# =====================================================
st.set_page_config(
    page_title="Employee Promotion App",
    page_icon="🏢",
    layout="wide"
)

# =====================================================
# 🏢 HEADER
# =====================================================
st.title("🏢 Employee Promotion Prediction App")
st.markdown("""
Selamat datang di **Employee Promotion App**!  
Gunakan sidebar di kiri untuk menavigasi ke:
- 📊 Dashboard
- 🔮 Prediction & Rekomendasi
- 🧠 Model Analysis
""")

# =====================================================
# ⚙️ LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    with open("models/rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    return model, feature_columns

model, feature_columns = load_model()
