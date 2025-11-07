"""
Employee Promotion Prediction App
Main Streamlit application
"""
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Employee Promotion Prediction",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Theme and Professional Look
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #f0f0f0;
    }
    
    /* Main Header Styling */
    .main-header {
        font-size: 2.5rem;
        color: #8b5cf6;
        text-align: center;
        margin-bottom: 2rem;
        padding-top: 1rem;
        font-weight: 700;
    }
    
    /* Card Styling for Model Information */
    .metric-card {
        background-color: #1f2a37;
        padding: 1.5rem;
        border-radius: 12px; 
        border-left: 5px solid #38bdf8; 
        color: #f0f0f0;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.4);
        margin-top: 1.5rem;
    }
    
    /* Ensure general text colors are light */
    h1, h2, h3, h4, p, label, .stText {
        color: #f0f0f0 !important;
    }
    
    /* Styling for Streamlit's internal dividers (st.markdown("---")) */
    .st-emotion-cache-1cyp68v { 
        border-top: 1px solid #38424f; 
        margin: 2rem 0;
    }
    
    /* Table styling inside the card for better look */
    .metric-card table {
        width: 100%;
        border-collapse: collapse;
    }
    .metric-card th, .metric-card td {
        padding: 0.5rem;
        text-align: left;
        border-bottom: 1px solid #38424f;
    }
    .metric-card th {
        color: #a3a3a3; /* Lighter color for column headers */
        font-weight: 400;
    }
    
</style>
""", unsafe_allow_html=True)

# Main Title
st.markdown('<h1 class="main-header">Employee Promotion Prediction Dashboard</h1>', unsafe_allow_html=True)

st.markdown('<h2 class="metric-card">Welcome!</h2>', unsafe_allow_html=True)

st.markdown("""
Halo! Selamat datang di sistem prediksi promosi karyawan. Aplikasi ini memanfaatkan model *machine learning* untuk memberikan penilaian kelayakan promosi berdasarkan kriteria kinerja dan data historis.
""")

# --- Pages Available Section (Using Columns for Structure) ---
st.header("✨ Navigasi Fitur Utama")
st.markdown("""
Silakan gunakan menu di sidebar untuk menjelajahi fungsionalitas aplikasi.
""")

# Use columns to present features clearly
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 Dashboard")
    st.write("Visualisasikan **Key Performance Indicators (KPI)**, metrik rata-rata karyawan, dan ringkasan performa model.")

with col2:
    st.markdown("### 🔮 Predictions")
    st.write("Unggah data baru atau masukkan detail karyawan secara manual untuk mendapatkan hasil prediksi promosi **secara *real-time***.")
    
with col3:
    st.markdown("### 📈 Model Analysis")
    st.write("Analisis mendalam mengenai **akurasi, presisi, recall**, serta pemeriksaan kesetaraan (*fairness*) model.")

st.markdown("---") # Visual divider

# --- Getting Started Section ---
st.header("🚀 Panduan Cepat")
st.markdown("""
Untuk memulai proses prediksi, Anda dapat langsung menuju halaman **Predictions** melalui sidebar. Pastikan data yang dimasukkan akurat untuk hasil prediksi yang optimal.
""")


# --- Model Information Section (Framed in a Card) ---
# Menggunakan st.container() untuk membingkai informasi model dengan custom class (metric-card)
st.markdown('<div class="metric-card">', unsafe_allow_html=True)
st.subheader("📝 Detail Informasi Model")

# Use a table for neat, aligned presentation of metrics
st.markdown("""
| Metrik | Nilai | Catatan |
| :--- | :--- | :--- |
| **Tipe Model** | Logistic Regression | Dipilih karena interpretasi yang baik. |
| **PR-AUC** | 0.350 | Area di bawah kurva Precision-Recall. |
| **Akurasi** | 0.544 | Tingkat kebenaran prediksi secara keseluruhan. |
| **Threshold (Kalibrasi)** | 0.209 | Nilai ambang batas untuk klasifikasi promosi. |
""")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<br>
""") 
st.markdown("---")
