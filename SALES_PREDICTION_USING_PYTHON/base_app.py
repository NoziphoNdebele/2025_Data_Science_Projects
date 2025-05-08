import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

# Load dataset
BASE_DIR = os.path.dirname(__file__)

@st.cache_data
def load_dataset():
    try:
        data_path = os.path.join(BASE_DIR, "cleaned_advertising.csv")
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

df = load_dataset()

if df.empty or 'TV' not in df.columns:
    st.error("❌ Dataset not loaded properly or missing expected columns ['TV', 'Sales'].")
    st.stop()


@st.cache_resource
def load_models():
    try:
        BASE_DIR = os.path.dirname(__file__)
        with open(os.path.join(BASE_DIR, 'tuned_decision_tree_regressor.pkl'), 'rb') as file:
            dt_model = pickle.load(file)
        with open(os.path.join(BASE_DIR, 'tuned_random_forest_regressor.pkl'), 'rb') as file:
            rf_model = pickle.load(file)
        return dt_model, rf_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

dt_model, rf_model = load_models()


# Linear Regression for baseline
lr_model = LinearRegression()
lr_model.fit(df[['TV']], df['Sales'])

# Custom CSS for pale blue background
st.markdown(
    """
    <style>
    body {
        background-color: #e6f2ff;
    }
    .stApp {
        background-color: #e6f2ff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.title("📊 Sales Prediction App")
st.markdown("**Project by: Nozipho Sithembiso Ndebele**")
st.markdown("This app predicts product sales based on TV advertising spend using different regression models.")

# Sidebar for navigation
st.sidebar.header("Choose a Model")
model_choice = st.sidebar.selectbox("Select Model", 
                                    ("Linear Regression", "Tuned Decision Tree Regressor", "Tuned Random Forest Regressor"))

# Input for prediction
tv_input = st.slider("Enter TV advertising spend ($)", 0.0, 300.0, 100.0, step=1.0)

# Predict
if model_choice == "Linear Regression":
    prediction = lr_model.predict(np.array([[tv_input]]))[0]
elif model_choice == "Tuned Decision Tree Regressor":
    prediction = dt_model.predict(np.array([[tv_input]]))[0]
else:
    prediction = rf_model.predict(np.array([[tv_input]]))[0]

st.subheader(f"Predicted Sales: **{prediction:.2f} units**")

# Show dataset
if st.checkbox("Show Raw Data"):
    st.write(df.head())

# Model Evaluation
st.subheader("📈 Model Performance (After Tuning)")
eval_data = {
    "Model": ["Tuned Decision Tree", "Tuned Random Forest"],
    "MSE": [5.9264, 5.3441],
    "RMSE": [2.4344, 2.3117],
    "R² Score": [0.8082, 0.8271]
}
st.dataframe(pd.DataFrame(eval_data))

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Nozipho Sithembiso Ndebele")

