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


# Load model and scaler
with open(os.path.join(BASE_DIR, 'best_movie_rating_model.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

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
    - Predict movie ratings using Random Forest
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

    # Input features
    duration = st.number_input("Duration (minutes)", min_value=30, max_value=240, value=120)
    votes = st.number_input("Number of Votes", min_value=100, value=10000)
    genre_avg = st.number_input("Genre Average Rating (0.0 - 10.0)", min_value=0.0, max_value=10.0, value=6.5)
    director_avg = st.number_input("Director Average Rating (0.0 - 10.0)", min_value=0.0, max_value=10.0, value=6.8)
    actor1_avg = st.number_input("Actor 1 Average Rating", min_value=0.0, max_value=10.0, value=6.5)
    actor2_avg = st.number_input("Actor 2 Average Rating", min_value=0.0, max_value=10.0, value=6.0)
    actor3_avg = st.number_input("Actor 3 Average Rating", min_value=0.0, max_value=10.0, value=5.5)
    genre_encoded = st.number_input("Genre Encoded (e.g., Action=0, Comedy=1...)", min_value=0, value=0)
    director_encoded = st.number_input("Director Encoded (unique ID)", min_value=0, value=100)

    input_data = pd.DataFrame({
        'Duration': [duration],
        'Votes': [votes],
        'Genre_Average_Rating': [genre_avg],
        'Director_Average_Rating': [director_avg],
        'Actor 1_Average_Rating': [actor1_avg],
        'Actor 2_Average_Rating': [actor2_avg],
        'Actor 3_Average_Rating': [actor3_avg],
        'Genre_encoded': [genre_encoded],
        'Director_encoded': [director_encoded],
    })

    scaled_input = scaler.transform(input_data)

    if st.button("Predict"):
        rating_pred = model.predict(scaled_input)[0]
        st.success(f"🎬 Predicted IMDb Rating: **{rating_pred:.2f}**")

# Tab 4: Model Evaluation
elif selected_tab == "Model Evaluation":
    st.title("📈 Model Evaluation")
    st.markdown("Visualize how well the model performs on test data.")

    try:
        eval_df = pd.read_csv(os.path.join(BASE_DIR, "model_predictions.csv"))

        st.subheader("Model Performance Metrics")
        st.dataframe(eval_df)

        st.subheader("Model Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Model', y='RMSE', data=eval_df, ax=ax)
        ax.set_title('Model Comparison by RMSE')
        st.pyplot(fig)

    except FileNotFoundError:
        st.warning("model_predictions.csv not found. Please add evaluation results.")

# Tab 5: Conclusion
elif selected_tab == "Conclusion":
    st.title("Conclusion")
    st.markdown("""
    This Streamlit app showcases a movie rating prediction model using Random Forest.

    Built with:
    - Machine Learning
    - Scikit-learn
    - Streamlit
    - Pandas & Seaborn
    """)
    st.balloons()
