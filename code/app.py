import streamlit as st
import re
import string
import pickle
import numpy as np
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.sparse import hstack

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Fake News Detection",
    layout="centered"
)

# -----------------------------
# Custom CSS (clean, minimal)
# -----------------------------
st.markdown("""
<style>
h1 {
    text-align: center;
    font-size: 3rem;
    margin-bottom: 0.2rem;
}
.subtitle {
    text-align: center;
    font-size: 1.1rem;
    color: #9ca3af;
    margin-bottom: 2.5rem;
}
.step {
    color: #22c55e;
    font-size: 0.85rem;
    letter-spacing: 1px;
    margin-bottom: 0.3rem;
}
textarea {
    border-radius: 14px !important;
}
.stButton button {
    width: 100%;
    border-radius: 12px;
    padding: 0.7rem;
    font-size: 1rem;
}
.result-box {
    padding: 1.2rem;
    border-radius: 14px;
    margin-top: 1.5rem;
    font-size: 1.05rem;
}
.real {
    background: rgba(34,197,94,0.15);
    border: 1px solid rgba(34,197,94,0.4);
}
.fake {
    background: rgba(239,68,68,0.15);
    border: 1px solid rgba(239,68,68,0.4);
}
.neutral {
    background: rgba(234,179,8,0.15);
    border: 1px solid rgba(234,179,8,0.4);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("<h1>Fake News Detection</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Identify potentially misleading news using advanced NLP analysis.</div>",
    unsafe_allow_html=True
)

# -----------------------------
# Load models
# -----------------------------
BASE_DIR = Path(__file__).parent

with open(BASE_DIR / "tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open(BASE_DIR / "logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

analyzer = SentimentIntensityAnalyzer()

# -----------------------------
# Utilities
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def validate_input(text):
    words = text.split()
    sentences = re.split(r"[.!?]", text)

    if len(text) < 50:
        return "Text must contain at least 50 characters."
    if len(words) < 5:
        return "Text must contain at least 5 words."
    if len([s for s in sentences if s.strip()]) < 1:
        return "Text must contain at least one complete sentence."
    return None

# -----------------------------
# Step 1 UI
# -----------------------------
st.markdown("<div class='step'>STEP 1</div>", unsafe_allow_html=True)
st.subheader("Paste Article")
st.caption("Paste the full news article below for analysis.")

article = st.text_area(
    "",
    placeholder="Paste a real news article here...",
    height=220
)

# -----------------------------
# Analyze
# -----------------------------
if st.button("Continue →"):
    error = validate_input(article)

    if error:
        st.warning(error)
    else:
        cleaned = clean_text(article)

        tfidf_vec = tfidf.transform([cleaned])
        sentiment = analyzer.polarity_scores(cleaned)
        sentiment_vec = np.array([[sentiment["compound"]]])

        X = hstack([tfidf_vec, sentiment_vec])

        proba = model.predict_proba(X)[0]
        confidence = float(np.max(proba))
        prediction = model.classes_[np.argmax(proba)]

        if confidence < 0.65:
            st.markdown(
                "<div class='result-box neutral'><b>Inconclusive</b><br>"
                "The input does not contain enough strong linguistic signals to make a reliable prediction.</div>",
                unsafe_allow_html=True
            )
        elif prediction == 1:
            st.markdown(
                f"<div class='result-box real'><b>Likely Real News</b><br>"
                f"Model confidence: {confidence:.2%}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result-box fake'><b>Likely Fake News</b><br>"
                f"Model confidence: {confidence:.2%}</div>",
                unsafe_allow_html=True
            )