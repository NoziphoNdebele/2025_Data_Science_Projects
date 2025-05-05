import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Movie Rating Predictor", layout="wide")

# Load model and scaler
with open('best_movie_rating_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load dataset for EDA (optional)
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_IMDb_Movies_India.csv")  # Update path if needed

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
    
    st.subheader("Sample Data")
    st.dataframe(df.head())

    st.subheader("Rating Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Rating'], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

elif selected_tab == "Predict Rating":
    st.title("🎯 Predict Movie Rating")

    st.markdown("Enter movie features below to predict the IMDb rating:")

    # Collect all features that the model was trained on
    duration = st.number_input("Duration (minutes)", min_value=30, max_value=240, value=120)
    votes = st.number_input("Number of Votes", min_value=100, value=10000)
    genre_avg = st.number_input("Genre Average Rating (0.0 - 10.0)", min_value=0.0, max_value=10.0, value=6.5)
    director_avg = st.number_input("Director Average Rating (0.0 - 10.0)", min_value=0.0, max_value=10.0, value=6.8)
    actor1_avg = st.number_input("Actor 1 Average Rating", min_value=0.0, max_value=10.0, value=6.5)
    actor2_avg = st.number_input("Actor 2 Average Rating", min_value=0.0, max_value=10.0, value=6.0)
    actor3_avg = st.number_input("Actor 3 Average Rating", min_value=0.0, max_value=10.0, value=5.5)
    genre_encoded = st.number_input("Genre Encoded (e.g., Action=0, Comedy=1...)", min_value=0, value=0)
    director_encoded = st.number_input("Director Encoded (unique ID)", min_value=0, value=100)

    # Arrange input data as per training format
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
        # Load the evaluation data
        eval_df = pd.read_csv("model_predictions.csv")
        
        # Show the evaluation results
        st.subheader("Model Performance Metrics")
        st.dataframe(eval_df)

        # Optional: You can also visualize it with a bar plot
        st.subheader("Model Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Model', y='RMSE', data=eval_df, ax=ax)
        ax.set_title('Model Comparison by RMSE')
        st.pyplot(fig)

    except FileNotFoundError:
        st.warning("model_predictions.csv not found. Please add evaluation results.")

# Tab 5: Conclusion
elif selected_tab == "Conclusion":
    st.title("📌 Conclusion")
    st.markdown("""
    This Streamlit app showcases a movie rating prediction model using Random Forest.

    ✅ Built with:
    - Machine Learning
    - Scikit-learn
    - Streamlit
    - Pandas & Seaborn

    📥 You can improve the model further by adding more features (e.g., director, actors, release year) or tuning hyperparameters!
    """)

    st.balloons()
