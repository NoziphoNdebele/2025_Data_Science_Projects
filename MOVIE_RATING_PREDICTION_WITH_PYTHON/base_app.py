import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(__file__)

# Set page config
st.set_page_config(page_title="Movie Rating Predictor", layout="wide")

# Apply custom CSS for pale red background
st.markdown(
    """
    <style>
        .stApp {
            background-color: #ffe6e6;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load model
with open(os.path.join(BASE_DIR, 'tuned_xgboost_model.pkl'), 'rb') as f:
    model = pickle.load(f)

# Load dataset for EDA
@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE_DIR, "cleaned_IMDb_Movies_India.csv"))

df = load_data()

# Sidebar
st.sidebar.title("Navigation")
tabs = ["Overview", "EDA", "Predict Rating", "Model Evaluation", "Conclusion"]
selected_tab = st.sidebar.radio("Go to", tabs)

# Tab 1: Overview
if selected_tab == "Overview":
    st.title("🎬 Movie Rating Prediction App")
    st.markdown("""
    This app uses machine learning to predict IMDb movie ratings based on features like genre, duration, budget, and more.
    
    **Key Features:**
    - Exploratory Data Analysis (EDA)
    - Predict movie ratings using XGBoost
    - Interactive inputs for new predictions
    - Visual evaluation of model performance
    """)

# Tab 2: EDA
elif selected_tab == "EDA":
    st.title("🔍 Exploratory Data Analysis")

    st.subheader("Rating Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Rating'], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Tab 3: Predict Rating
elif selected_tab == "Predict Rating":
    st.title("🎯 Predict Movie Rating")
    st.markdown("Enter movie features below to predict the IMDb rating:")

    # Input features matching the training data
    year = st.number_input("Year", min_value=1900, max_value=2025, value=2020)
    votes = st.number_input("Number of Votes", min_value=100, value=10000)
    genre_avg = st.number_input("Genre Average Rating (0.0 - 10.0)", min_value=0.0, max_value=10.0, value=6.5)
    director_avg = st.number_input("Director Average Rating (0.0 - 10.0)", min_value=0.0, max_value=10.0, value=6.8)
    actor1_avg = st.number_input("Actor 1 Average Rating", min_value=0.0, max_value=10.0, value=6.5)
    actor2_avg = st.number_input("Actor 2 Average Rating", min_value=0.0, max_value=10.0, value=6.0)
    actor3_avg = st.number_input("Actor 3 Average Rating", min_value=0.0, max_value=10.0, value=5.5)

    # Prepare the input data for prediction (matching the feature set in training)
    input_data = pd.DataFrame({
        'Year': [year],
        'Votes': [votes],
        'Genre_Average_Rating': [genre_avg],
        'Director_Average_Rating': [director_avg],
        'Actor1_Average_Rating': [actor1_avg],
        'Actor2_Average_Rating': [actor2_avg],
        'Actor3_Average_Rating': [actor3_avg],
    })

    if st.button("Predict"):
        rating_pred = model.predict(input_data)[0]
        st.success(f"🎬 Predicted IMDb Rating: **{rating_pred:.2f}**")

# Tab 4: Model Evaluation
elif selected_tab == "Model Evaluation":
    st.title("📈 Model Evaluation")
    st.markdown("Visualize how well the models perform on test data.")

    # Displaying results for Tuned Random Forest and Tuned XGBoost models
    st.subheader("Tuned Random Forest Evaluation:")

    st.write("**MAE:** 0.4008")
    st.write("**MSE:** 0.3625")
    st.write("**RMSE:** 0.6020")
    st.write("**R²:** 0.7975")

    st.subheader("Tuned XGBoost Evaluation:")
   
    st.write("**MAE:** 0.3937")
    st.write("**MSE:** 0.3283")
    st.write("**RMSE:** 0.5730")
    st.write("**R²:** 0.8166")

    # add a bar plot comparison
    model_comparison = pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost'],
        'RMSE': [0.6020, 0.5730],
    })

    st.subheader("Model Comparison by RMSE")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Model', y='RMSE', data=model_comparison, ax=ax)
    ax.set_title('Model Comparison by RMSE')
    st.pyplot(fig)


# Tab 5: Conclusion
elif selected_tab == "Conclusion":
    st.title("Conclusion")
    st.markdown("""
    This Streamlit app showcases a movie rating prediction model using XGBoost.

    Built with:
    - Machine Learning
    - Scikit-learn
    - Streamlit
    - Pandas & Seaborn
    """)
    st.balloons()
