import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard", layout="wide")

@st.cache_data
def load_real_data():
    base_path = os.path.dirname(os.path.dirname(__file__))  # naik satu folder dari /pages
    data_path = os.path.join(base_path, "data", "employee_promotion_dataset.csv")
    df = pd.read_csv(data_path)
    return df

def show_dashboard():
    st.title("📊 Employee Promotion Dashboard")
    
    df = load_real_data()
    st.dataframe(df.head())

    st.subheader("Distribusi Target")
    fig, ax = plt.subplots()
    df['Promotion_Eligible'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

    st.subheader("Rata-rata nilai per departemen")
    avg_score = df.groupby('Position_Level')['Performance_Score'].mean().sort_values(ascending=False)
    st.bar_chart(avg_score)

show_dashboard()

