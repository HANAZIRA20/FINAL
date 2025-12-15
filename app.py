# ============================================================
# IMPORT LIBRARY
# ============================================================
import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Netflix Movie vs TV Show Classification",
    page_icon="🎬",
    layout="wide"
)

# ============================================================
# HEADER
# ============================================================
st.markdown("<h1 style='text-align:center;'>🎬 Netflix Movie vs TV Show Classification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Decision Tree & Random Forest | Data Mining Project</p>", unsafe_allow_html=True)
st.divider()

# ============================================================
# LOAD DATASET
# ============================================================
DATA_PATH = "netflix_titles.csv"

if not os.path.exists(DATA_PATH):
    st.error("❌ Dataset netflix_titles.csv tidak ditemukan.")
    st.stop()

df = pd.read_csv(DATA_PATH)
st.success("✅ Dataset berhasil dimuat")

# ============================================================
# TARGET → MOVIE (1) VS TV SHOW (0)
# ============================================================
df["label"] = df["type"].map({"Movie": 1, "TV Show": 0})

# ============================================================
# HANDLE MISSING VALUE
# ============================================================
df = df.fillna("Unknown")

# ============================================================
# FEATURE ENGINEERING (AMAN, TANPA LEAKAGE)
# ============================================================
df["description_length"] = df["description"].apply(lambda x: len(str(x)))
df["num_genres"] = df["listed_in"].apply(lambda x: len(str(x).split(",")))
df["num_countries"] = df["country"].apply(lambda x: len(str(x).split(",")))

# Fitur aman
features = [
    "release_year",
    "description_length",
    "num_genres",
    "num_countries"
]

df_proc = df[features + ["label"]]

# ============================================================
# DATA OVERVIEW
# ============================================================
st.subheader("📊 1. Data Overview")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**5 Data Teratas**")
    st.dataframe(df_proc.head(), use_container_width=True)

with col2:
    info_df = pd.DataFrame({
        "Kolom": df_proc.columns,
        "Tipe Data": df_proc.dtypes.astype(str),
        "Missing": df_proc.isnull().sum()
    })
    st.markdown("**Informasi Dataset**")
    st.dataframe(info_df, use_container_width=True)

st.divider()

# ============================================================
# TARGET DISTRIBUTION
# ============================================================
st.subheader("🎯 2. Distribusi Target (Movie vs TV Show)")

col1, col2 = st.columns(2)

with col1:
    st.dataframe(df["label"].value_counts())

with col2:
    fig, ax = plt.subplots(figsize=(3.5,2.5))
    df["label"].value_counts().plot(kind="bar", ax=ax, color=["green", "red"])
    ax.set_xlabel("Label (0 = TV Show, 1 = Movie)")
    ax.set_ylabel("Jumlah")
    st.pyplot(fig)

st.divider()

# ============================================================
# SPLIT DATA
# ============================================================
X = df_proc.drop(columns=["label"])
y = df_proc["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

st.subheader("📂 3. Pembagian Data")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Data", df_proc.shape[0])
with col2:
    st.metric("Data Training", X_train.shape[0])
with col3:
    st.metric("Data Testing", X_test.shape[0])

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
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# ============================================================
# EVALUASI MODEL
# ============================================================
st.subheader("🤖 4. Evaluasi Model")

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
if hasattr(model, "feature_importances_"):
    st.subheader("📌 Feature Importance")

    fig_imp, ax_imp = plt.subplots(figsize=(4,3))
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances.sort_values().plot(kind="barh", ax=ax_imp, color="teal")
    ax_imp.set_title("Feature Importance")
    st.pyplot(fig_imp)

st.divider()

# ============================================================
# PRECISION-RECALL CURVE
# ============================================================
st.subheader("📈 Precision-Recall Curve")

y_scores = model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
avg_precision = average_precision_score(y_test, y_scores)

fig_pr, ax_pr = plt.subplots(figsize=(4,3))
ax_pr.plot(recall, precision, color="purple", linewidth=2)
ax_pr.set_title(f"PR Curve (AP = {avg_precision:.2f})")
ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
ax_pr.grid(True)
st.pyplot(fig_pr)

st.divider()

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    "<p style='text-align:center;font-size:12px;'>Data Mining Project | Streamlit</p>",
    unsafe_allow_html=True
)
