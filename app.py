import streamlit as st
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Title
st.title("📰 Fake News Detection App")

st.write("Enter a news article below to check whether it is Fake or Real.")

# Input
user_input = st.text_area("Enter News Text")

# Button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        vec = vectorizer.transform([user_input])
        result = model.predict(vec)

        if result[0] == 1:
            st.success("✅ This is Real News")
        else:
            st.error("❌ This is Fake News")