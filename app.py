import streamlit as st
import joblib

st.title("📺 YouTube Spam Detector")
st.write("Paste a YouTube comment and check if it's Spam or Not Spam.")

model = joblib.load("models/spam.pkl")

comment = st.text_area("Enter a comment:")
if st.button("Predict"):
    prediction = model.predict([comment])[0]
    if prediction == 1:
        st.error("🚨 Spam Comment Detected!")
    else:
        st.success("✅ Not Spam")
