import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Load model and vectorizer
model = load_model("sustainability_model.h5")
vectorizer = joblib.load("vectorizer.pkl")

# UI
st.title("🌱 Sustainable Product Classifier")
st.markdown("Classify if a product is **eco-friendly** based on its description. 🌍")

user_input = st.text_area("Enter product description:", "")

if st.button("Check Sustainability"):
    if user_input.strip() != "":
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec.toarray())[0][0]

        if prediction >= 0.5:
            st.success("✅ This product is **Eco-Friendly** 💚")
        else:
            st.error("❌ This product is **Not Eco-Friendly** 💔")
    else:
        st.warning("Please enter a description to check.")
