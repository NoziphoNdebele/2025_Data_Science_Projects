import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained models
@st.cache_resource
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# Evaluation Function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))
    st.text(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred):.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_title("Confusion Matrix")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    # ROC Curve
    fig_roc, ax_roc = plt.subplots()
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax_roc)
    ax_roc.set_title("ROC Curve")
    st.pyplot(fig_roc)

    # Precision-Recall Curve
    fig_pr, ax_pr = plt.subplots()
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax_pr)
    ax_pr.set_title("Precision-Recall Curve")
    st.pyplot(fig_pr)

# App Title
st.title("💳 Credit Card Fraud Detection")
st.markdown("Project by **Nozipho Sithembiso Ndebele**")

# Sidebar for model selection
model_choice = st.sidebar.selectbox(
    "Choose a model:",
    ("Tuned Random Forest", "Tuned XGBoost", "Tuned CatBoost")
)

model_paths = {
    "Tuned Random Forest": "rf_best_model.pkl",
    "Tuned XGBoost": "xgb_best_model.pkl",
    "Tuned CatBoost": "cat_best_model.pkl"
}

# Load selected model
model = load_model(model_paths[model_choice])

# Upload test dataset
uploaded_file = st.file_uploader("Upload a test CSV file with same structure as training data:", type="csv")

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    if 'Class' in test_data.columns:
        X_test = test_data.drop('Class', axis=1)
        y_test = test_data['Class']
        st.success("✅ Data loaded successfully. Running evaluation...")
        evaluate_model(model, X_test, y_test)
    else:
        st.error("❌ The dataset must include a 'Class' column as ground truth.")
else:
    st.info("Please upload a test dataset to continue.")

