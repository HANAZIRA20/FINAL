# ============================================================
# INPUT MANUAL (SIMPLE VERSION)
# ============================================================
st.subheader("📝 5. Input Manual untuk Prediksi (Versi Simple)")

user_input = {}

col1, col2 = st.columns(2)

with col1:
    user_input["tenure"] = st.number_input(
        "Tenure (Lama Berlangganan / bulan)",
        min_value=0,
        max_value=100,
        value=12
    )

    user_input["MonthlyCharges"] = st.number_input(
        "Monthly Charges (Tagihan Bulanan)",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

    user_input["Contract"] = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

with col2:
    user_input["InternetService"] = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    user_input["PaymentMethod"] = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

    user_input["TechSupport"] = st.selectbox(
        "Tech Support",
        ["Yes", "No"]
    )

# Convert ke DataFrame
user_df = pd.DataFrame([user_input])

# Encode manual input agar cocok dengan model
user_encoded = pd.get_dummies(user_df)
user_encoded = user_encoded.reindex(columns=X.columns, fill_value=0)
