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

st.markdown("## Welcome!")

st.markdown("""
This application uses machine learning to predict employee promotion eligibility.
""")

# --- Pages Available Section (Using Columns for Structure) ---
st.header("📑 Pages Available")
st.markdown("""
Navigate using the sidebar to explore:
""")

# Use columns to present features clearly
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 Dashboard")
    st.write("View KPIs, model performance, and prediction tables.")

with col2:
    st.markdown("### 🔮 Predictions")
    st.write("Upload CSV for predictions.")
    
with col3:
    st.markdown("### 📈 Model Analysis")
    st.write("Detailed model evaluation and fairness analysis.")

st.markdown("---") # Visual divider

# --- Model Information Section (Framed in a Card) ---
st.subheader("📝 Model Information")

# Use a table for neat, aligned presentation of metrics
st.markdown("""
| Metrik | Nilai | Catatan |
| :--- | :--- | :--- |
| **Tipe Model** | Random Forest |
| **PR-AUC** | 0.350 | 
| **Accuracy** | 0.544 | 
| **Threshold** | 0.209 |
""")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<br>
""") 
st.markdown("---")
