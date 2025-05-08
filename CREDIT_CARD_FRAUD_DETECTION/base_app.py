import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Load trained models
@st.cache_resource
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# ------------------------------
# Evaluation Function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    roc_score = roc_auc_score(y_test, y_pred)

    # 💬 Summary Verdict
    st.subheader("🧠 Model Verdict")
    if roc_score > 0.85:
        st.success("🛡️ Excellent model performance. It can reliably detect fraudulent transactions.")
    elif roc_score > 0.70:
        st.warning("⚠️ Fair performance. It catches most fraud but may miss a few cases.")
    else:
        st.error("❌ Poor performance. Use with caution — it may miss many frauds.")

    # 📊 Classification Metrics
    st.subheader("📊 Classification Report")
    st.code(classification_report(y_test, y_pred), language='text')
    st.metric(label="ROC-AUC Score", value=f"{roc_score:.4f}")

    # 📘 Metric Explanations
    st.markdown("""
    ### ℹ️ What the Metrics Mean
    - **Precision**: Of the transactions predicted as fraud, how many actually were fraud.
    - **Recall**: Of the actual fraud cases, how many were correctly detected.
    - **F1-Score**: Balance between precision and recall.
    - **ROC-AUC**: Overall ability of the model to distinguish fraud from non-fraud (closer to 1.0 is better).
    """)

    # 🔍 Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_title("🔍 Confusion Matrix")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    # 📈 ROC Curve
    fig_roc, ax_roc = plt.subplots()
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax_roc)
    ax_roc.set_title("📈 ROC Curve")
    st.pyplot(fig_roc)

    # 📉 Precision-Recall Curve
    fig_pr, ax_pr = plt.subplots()
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax_pr)
    ax_pr.set_title("📉 Precision-Recall Curve")
    st.pyplot(fig_pr)

    # 📊 Pie Chart of Prediction Summary
    pred_counts = pd.Series(y_pred).value_counts().rename({0: 'Not Fraud', 1: 'Fraud'})
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(pred_counts, labels=pred_counts.index, autopct='%1.1f%%', startangle=90)
    ax_pie.set_title("Predicted Fraud vs. Not Fraud")
    st.pyplot(fig_pie)

# ------------------------------
# Streamlit App UI
st.set_page_config(page_title="Fraud Detection App", layout="wide")

# Set light grey background
st.markdown(
    """
    <style>
    .stApp {
        background-color: #D3D3D3;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown("A machine learning-powered tool to evaluate models trained to detect fraudulent credit card transactions.")
st.markdown("**Project by: Nozipho Sithembiso Ndebele**")

# Sidebar for model selection
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose a trained model:",
    ("Tuned Random Forest", "Tuned XGBoost", "Tuned CatBoost")
)

model_paths = {
    "Tuned Random Forest": "rf_best_model.pkl",
    "Tuned XGBoost": "xgb_best_model.pkl",
    "Tuned CatBoost": "cat_best_model.pkl"
}

# Load selected model
with st.spinner("Loading model..."):
    model = load_model(model_paths[model_choice])
st.sidebar.success(f"{model_choice} loaded.")

# Load built-in test data from project directory
st.subheader("📂 Evaluating Preloaded Test Dataset")
try:
    test_data = pd.read_csv("test_data.csv")  # Ensure this file exists in the app folder

    if 'Class' in test_data.columns:
        X_test = test_data.drop('Class', axis=1)
        y_test = test_data['Class']
        st.success("✅ Preloaded test data found. Running model evaluation...")
        evaluate_model(model, X_test, y_test)
    else:
        st.error("❌ 'Class' column is missing in the internal test dataset.")
except Exception as e:
    st.error("❌ Failed to load test data. Make sure 'test_data.csv' is in the app directory.")
    st.exception(e)
