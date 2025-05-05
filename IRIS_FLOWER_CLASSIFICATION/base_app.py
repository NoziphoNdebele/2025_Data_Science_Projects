import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Page configuration
st.set_page_config(page_title="Iris Classifier 🌸", layout="centered")

# Load models and label encoder
@st.cache_resource
def load_models():
    try:
        with open("best_knn_model.pkl", "rb") as f:
            knn = pickle.load(f)
        with open("best_svm_model.pkl", "rb") as f:
            svm = pickle.load(f)
        with open("label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        return {"KNN": knn, "SVM": svm}, le
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return {}, None

models, le = load_models()

# Dummy sample DataFrame for display
df = pd.DataFrame({
    "sepal length (cm)": [5.1, 7.0, 6.3],
    "sepal width (cm)": [3.5, 3.2, 3.3],
    "petal length (cm)": [1.4, 4.7, 6.0],
    "petal width (cm)": [0.2, 1.4, 2.5],
    "species": ["setosa", "versicolor", "virginica"]
})

# Dummy results (use your actual values here)
results = {
    "KNN": 0.96,
    "SVM": 0.98
}

# App title
st.title("🌸 Iris Flower Classification App")
st.markdown("Created by **Nozipho Sithembiso Ndebele**")

# Dataset Preview
st.header("🔍 Sample Dataset")
st.dataframe(df)

# Model Accuracies
st.header("📊 Model Accuracies")
fig, ax = plt.subplots()
model_names = list(results.keys())
acc_values = list(results.values())
ax.barh(model_names, acc_values, color='orchid')
for i, v in enumerate(acc_values):
    ax.text(v + 0.001, i, f"{v:.2%}", va='center')
ax.set_xlim(0.9, 1.0)
ax.set_xlabel("Accuracy")
st.pyplot(fig)

# Prediction Section
st.header("🌼 Predict Iris Species")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)

with col2:
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

selected_model_name = st.selectbox("Select Model", list(models.keys()))
selected_model = models[selected_model_name]

if st.button("Predict"):
    prediction = selected_model.predict(input_data)[0]
    predicted_label = le.inverse_transform([prediction])[0]
    st.success(f"🌷 Predicted species: **{predicted_label}** using **{selected_model_name}**")

# Footer
st.markdown("---")
st.caption("© 2025 Nozipho Sithembiso Ndebele")
