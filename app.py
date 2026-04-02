import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

# -------- LOAD DATA --------
df = pd.read_csv("data/deceptive-opinion.csv")
df = df[['text', 'deceptive', 'polarity']]
df.columns = ['review', 'label', 'polarity']
df['label'] = df['label'].map({'truthful': 0, 'deceptive': 1})
df['review'] = df['review'].str.lower()

# -------- FEATURES --------
fake_words = [
    'amazing','perfect','best','must','recommended',
    'guaranteed','100%','top','excellent','fantastic',
    'incredible','outstanding','ultimate','superb'
]
X_text = df['review']
X_extra = df['polarity'].map({'positive': 1, 'negative': 0}).values.reshape(-1,1)

df['fake_score'] = df['review'].apply(
    lambda x: sum(word in x for word in fake_words) * 2
)

X_extra2 = df['fake_score'].values.reshape(-1,1)

# -------- VECTORIZER --------
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1,2),
    token_pattern=r'(?u)\b\w+\b|!+'
)

X_text_vec = vectorizer.fit_transform(X_text)

X_combined = hstack([X_text_vec, X_extra, X_extra2])

y = df['label']

# -------- MODEL --------
model = LogisticRegression(max_iter=1000)
model.fit(X_combined, y)

# -------- EXPLANATION FUNCTION --------
def explain_review(text):
    reasons = []
    words = text.split()

    if len(words) != len(set(words)):
        reasons.append("Repeated words")

    if "!" in text:
        reasons.append("Excessive punctuation")

    if any(word in text for word in fake_words):
        reasons.append("Promotional language")

    if len(words) < 5:
        reasons.append("Very short review")

    return ", ".join(reasons) if reasons else "Looks natural"

# -------- STREAMLIT UI --------
st.set_page_config(page_title="Fake Review Detector", layout="centered")

st.title("🧠 Fake Review Detector")
st.write("Enter a review and check if it's Fake or Genuine")

user_input = st.text_area("✍️ Enter Review")

if st.button("Check Review"):

    if user_input.strip() == "":
        st.warning("Please enter a review")
    else:
        text = user_input.lower()

        review_vec = vectorizer.transform([text])
        extra_feature = np.array([[1]])
        fake_score = sum(word in text for word in fake_words) * 2
        extra_feature2 = np.array([[fake_score]])

        final_input = hstack([review_vec, extra_feature, extra_feature2])

        prediction = model.predict(final_input)
        prob = model.predict_proba(final_input)
        confidence = max(prob[0]) * 100

        if prediction[0] == 1:
            st.error(f"Fake ❌ (Confidence: {confidence:.2f}%)")
        else:
            st.success(f"Genuine ✅ (Confidence: {confidence:.2f}%)")

        reason = explain_review(text)
        st.info(f"Reason: {reason}")