import streamlit as st
import pickle

# ---------- CONFIG ----------
st.set_page_config(page_title="Resume Category Predictor", layout="centered")
st.title("📄 Resume Category Predictor")
st.markdown("Upload or paste your resume text and get the predicted job category.")

# ---------- LOAD PICKLE FILES CORRECTLY ----------
with open("svm_model.pkl", "rb") as model_file:
    svm_model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

with open("label_encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

# ---------- PREDICTION FUNCTION ----------
def predict_category(text):
    if not text.strip():
        return "Please enter some resume content."
    features = tfidf_vectorizer.transform([text])
    prediction = svm_model.predict(features)
    category = label_encoder.inverse_transform(prediction)
    return category[0]

# ---------- STREAMLIT UI ----------
input_option = st.radio("Select Input Method", ["✍️ Paste Resume Text", "📁 Upload Resume File (.txt)"])

resume_text = ""

if input_option == "✍️ Paste Resume Text":
    resume_text = st.text_area("Paste your resume content here:", height=300)
elif input_option == "📁 Upload Resume File (.txt)":
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
    if uploaded_file is not None:
        resume_text = uploaded_file.read().decode("utf-8")

if st.button("Predict Category"):
    if resume_text:
        category = predict_category(resume_text)
        st.success(f"🎯 Predicted Job Category: **{category}**")
    else:
        st.warning("Please provide resume content to predict.")

# ---------- FOOTER ----------
st.markdown("---")
st.caption("Made with ❤️ by Madhumita Das")
