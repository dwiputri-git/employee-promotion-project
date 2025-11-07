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

# Custom CSS for Dark Theme
st.markdown("""
<style>
    .stApp {
        background-color: #636573;
        color: #fafafa;
    }
    
    .main-header {
        font-size: 2.5rem;
        color: #8b5cf6; /* A brighter, contrasting purple for dark mode */
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Darker card background for contrast, and a bright accent border */
    .metric-card {
        background-color: #1f2a37; /* Slightly lighter than the main background */
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #38bdf8; /* Vibrant light blue/cyan accent */
        color: #f0f0f0; /* Ensure text inside the card is light */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); /* Subtle shadow for depth */
    }
    
    /* Ensure general text colors are light */
    h1, h2, h3, h4, p, label, .stText {
        color: #f0f0f0 !important;
    }
    
    /* Adjust Streamlit's native elements text color (e.g., Markdown blocks) */
    div[data-testid="stMarkdownContainer"] {
        color: #f0f0f0;
    }
    
    .prediction-table {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Welcome page
st.markdown('<h1 class="main-header">Employee Promotion Prediction</h1>', unsafe_allow_html=True)

st.markdown("""
## Welcome!

This application uses machine learning to predict employee promotion eligibility.

### 📊 Pages Available

Navigate using the sidebar to explore:

1. **📊 Dashboard** - View KPIs, model performance, and prediction tables
2. **🔮 Predictions** - Upload CSV or input data manually for predictions  
3. **📈 Model Analysis** - Detailed model evaluation and fairness analysis

### 🚀 Getting Started

Use the sidebar navigation to explore the different features of the application.

### 📝 Model Information

- **Model Type**: Logistic Regression
- **PR-AUC**: 0.350
- **Accuracy**: 0.544
- **Threshold**: 0.209 (calibrated)
""")
