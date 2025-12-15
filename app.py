# ============================================================
# IMPORT LIBRARY
# ============================================================
import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Telco Customer Churn",
    page_icon="📱",
    layout="wide"
)

# ============================================================
# HEADER
# ============================================================
st.markdown("<h1 style='text-align:center;'>📱 Telco Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Decision Tree & Random Forest | Data Mining Project</p>", unsafe_allow_html=True)
st.divider()

# ============================================================
# LOAD DATASET
# ============================================================
DATA_PATH = "Dataset Telco-Customer-Churn.csv"

if not os.path.exists(DATA_PATH):
    st.error("❌ Dataset tidak ditemukan.")
    st.stop()

df = pd.read_csv(DATA_PATH)
st.success("✅ Dataset berhasil dimuat")

# ============================================================
# FIX TARGET → BINARY
# ============================================================
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ============================================================
# HANDLE MISSING VALUE & NUMERIC CONVERSION
# ============================================================
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.fillna(df.median(numeric_only=True))
df = df.fillna(df.mode().iloc[0])

# ============================================================
# DATA OVERVIEW
# ============================================================
st.subheader("📊 1. Data Overview")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**5 Data Teratas**")
    st.dataframe(df.head(), use_container_width=True)

with col2:
    info_df = pd.DataFrame({
        "Kolom": df.columns,
        "Tipe Data": df.dtypes.astype(str),
        "Missing": df.isnull().sum()
    })
    st.markdown("**Informasi Dataset**")
    st.dataframe(info_df, use_container_width=True)

st.divider()

# ============================================================
# TARGET VARIABLE
# ============================================================
st.subheader("🎯 2. Target Variable (Churn)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Distribusi Target**")
    st.dataframe(df["Churn"].value_counts())

with col2:
    fig, ax = plt.subplots(figsize=(3.5,2.5))
    df["Churn"].value_counts().plot(kind="bar", ax=ax, color=["green", "red"])
    ax.set_xlabel("Churn (0 = Tidak Churn, 1 = Churn)")
    ax.set_ylabel("Jumlah")
    st.pyplot(fig)

st.divider()

# ============================================================
# PREPROCESSING
# ============================================================
st.subheader("⚙️ 3. Preprocessing Data")

df_proc = df.drop(columns=["customerID"], errors="ignore")
df_proc = df_proc.replace({"Yes": 1, "No": 0})

df_encoded = pd.get_dummies(df_proc, drop_first=True)

X = df_encoded.drop(columns=["Churn"])
y = df_encoded["Churn"]

st.write("🔍 Kolom fitur yang digunakan untuk prediksi:")
st.write(list(X.columns))

st.success("✅ Preprocessing selesai")
st.divider()

# ============================================================
# SPLIT DATA
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

st.subheader("📂 4. Pembagian Data")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Data", df_proc.shape[0])
with col2:
    st.metric("Data Training", X_train.shape[0])
with col3:
    st.metric("Data Testing", X_test.shape[0])

st.markdown("**Rasio:** 80% Training – 20% Testing")
st.divider()

# ============================================================
# MODEL SELECTION
# ============================================================
st.sidebar.header("⚙️ Pengaturan Model")

model_choice = st.sidebar.selectbox(
    "Pilih Model",
    ["Decision Tree", "Random Forest"]
)

if model_choice == "Decision Tree":
    model = DecisionTreeClassifier(random_state=42)
else:
    model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# ============================================================
# INPUT MANUAL (FORM)
# ============================================================
st.subheader("📝 5. Input Manual untuk Prediksi")

manual_cols = df_proc.drop(columns=["Churn"]).columns
user_input = {}

col1, col2 = st.columns(2)

with col1:
    for col in manual_cols[:len(manual_cols)//2]:
        if df_proc[col].dtype == "object":
            options = df_proc[col].astype(str).unique().tolist()
            user_input[col] = st.selectbox(col, options)
        else:
            user_input[col] = st.number_input(
                col,
                float(df_proc[col].min()),
                float(df_proc[col].max()),
                float(df_proc[col].mean())
            )

with col2:
    for col in manual_cols[len(manual_cols)//2:]:
        if df_proc[col].dtype == "object":
            options = df_proc[col].astype(str).unique().tolist()
            user_input[col] = st.selectbox(col, options)
        else:
            user_input[col] = st.number_input(
                col,
                float(df_proc[col].min()),
                float(df_proc[col].max()),
                float(df_proc[col].mean())
            )

user_df = pd.DataFrame([user_input])
user_encoded = pd.get_dummies(user_df)
user_encoded = user_encoded.reindex(columns=X.columns, fill_value=0)

st.subheader("🎯 6. Hasil Prediksi")

if st.button("🔍 Prediksi Sekarang"):
    manual_pred = model.predict(user_encoded)[0]
    manual_prob = model.predict_proba(user_encoded)[0][1]

    if manual_pred == 1:
        st.error(f"⚠️ Pelanggan diprediksi **CHURN** (Probabilitas: {manual_prob:.2f})")
    else:
        st.success(f"✅ Pelanggan diprediksi **TIDAK CHURN** (Probabilitas: {manual_prob:.2f})")

st.divider()

# ============================================================
# EVALUASI MODEL
# ============================================================
st.subheader("📊 7. Evaluasi Model")

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", f"{acc:.2f}")
    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

with col2:
    fig_cm, ax_cm = plt.subplots(figsize=(4,3))
    cm = confusion_matrix(y_test, y_pred)
    labels = [["TN", "FP"], ["FN", "TP"]]
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", ax=ax_cm, cbar=False)
    for i in range(2):
        for j in range(2):
            ax_cm.text(j + 0.5, i + 0.65, f"{cm[i, j]}", ha='center', va='center', color='black')
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

st.divider()

# ============================================================
# FEATURE IMPORTANCE
# ============================================================
st.subheader("📌 8. Feature Importance")

fig_imp, ax_imp = plt.subplots(figsize=(4,3))
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind="barh", ax=ax_imp, color="teal")
ax_imp.set_title("Feature Importance")
st.pyplot(fig_imp)

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown(
    "<p style='text-align:center;font-size:12px;'>Data Mining Project | Streamlit</p>",
    unsafe_allow_html=True
)
