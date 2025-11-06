# =====================================================
# 📊 DASHBOARD PAGE
# =====================================================
if menu == "📊 Dashboard":
    st.subheader("📊 Employee Promotion Dashboard")
    st.markdown("Berikut ringkasan performa promosi karyawan berdasarkan data dan prediksi model.")

    # Contoh: load data (gunakan data kamu sendiri)
    df = pd.read_csv("data/employee_promotion.csv")

    # Pastikan kolom 'Promotion_Eligible' ada
    total_employee = len(df)
    total_promoted = df['Promotion_Eligible'].sum() if 'Promotion_Eligible' in df.columns else 0
    promotion_rate = (total_promoted / total_employee * 100) if total_employee > 0 else 0

    # ===== Metric Cards =====
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👥 Total Employees", f"{total_employee:,}")
    with col2:
        st.metric("🏅 Predicted Promotions", f"{total_promoted:,}")
    with col3:
        st.metric("📈 Promotion Rate", f"{promotion_rate:.2f}%")

    # ===== Optional Visualization =====
    st.markdown("---")
    st.markdown("### 📊 Visualisasi Tambahan")

    colA, colB = st.columns(2)
    with colA:
        st.bar_chart(df['Promotion_Eligible'].value_counts(), use_container_width=True)
    with colB:
        if 'Age' in df.columns:
            st.line_chart(df[['Age']].sort_values('Age').reset_index(drop=True))
        else:
            st.info("Kolom 'Age' tidak ditemukan di dataset.")
