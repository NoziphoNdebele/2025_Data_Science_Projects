import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load trained model (ensure you have saved it as 'best_model.pkl')
model = joblib.load('best_logistic_model.pkl')

st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")

# Title and Description
st.title("🚢 Titanic Survival Prediction App")
st.markdown("""
Welcome! This app uses a machine learning model trained on Titanic passenger data to predict survival.
Enter passenger details below to see the prediction.
""")

# Sidebar: User input
st.sidebar.header("🧍 Passenger Information")

def user_input_features():
    pclass = st.sidebar.selectbox("Passenger Class (1 = Upper, 2 = Middle, 3 = Lower)", [1, 2, 3])
    sex = st.sidebar.selectbox("Sex", ['male', 'female'])
    age = st.sidebar.slider("Age", 0, 80, 30)
    sibsp = st.sidebar.slider("Number of Siblings/Spouses Aboard", 0, 8, 0)
    parch = st.sidebar.slider("Number of Parents/Children Aboard", 0, 6, 0)
    fare = st.sidebar.slider("Fare Paid", 0.0, 512.0, 50.0)
    embarked = st.sidebar.selectbox("Port of Embarkation", ['S', 'Q', 'C'])

    # Encode inputs
    sex = 0 if sex == 'male' else 1
    embarked_Q = True if embarked == 'Q' else False
    embarked_S = True if embarked == 'S' else False

    data = {
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked_Q': embarked_Q,
        'Embarked_S': embarked_S
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Prediction
st.subheader("📋 Passenger Details")
st.write(input_df)

prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("🔍 Prediction Result")
outcome = "Survived ✅" if prediction[0] == 1 else "Did Not Survive ❌"
st.markdown(f"**Prediction:** {outcome}")
st.markdown(f"**Confidence:** {np.max(prediction_proba) * 100:.2f}%")

# Optional: Add results from initial model evaluation
st.subheader("📊 Model Comparison Results")
results = {
    'Random Forest': [0.8212, 0.7917, 0.7703, 0.7808],
    'Logistic Regression': [0.8101, 0.7857, 0.7432, 0.7639],
    'Decision Tree': [0.7877, 0.7368, 0.7568, 0.7467]
}
results_df = pd.DataFrame(results, index=['Accuracy', 'Precision', 'Recall', 'F1-Score']).T
st.dataframe(results_df)

# Plot model comparison
fig, ax = plt.subplots(figsize=(10, 5))
results_df.plot(kind='bar', ax=ax)
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xticks(rotation=0)
st.pyplot(fig)

# About
st.subheader("📌 Insights & Conclusion")
st.markdown("""
- **Random Forest** and **Logistic Regression** models performed well, with accuracies above 81%.
- **Tuned Logistic Regression** slightly outperformed Random Forest after hyperparameter tuning.
- The model can help identify survival likelihood based on key features like age, sex, and class.

### 🚀 Future Work:
- Add more features (e.g., family size, cabin).
- Deploy using Docker or on the cloud (e.g., Streamlit Community Cloud).
- Expand to a full dashboard with SHAP visualizations for model explainability.
""")
