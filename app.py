import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/deceptive-opinion.csv")
    df = df[['text', 'deceptive']]
    df.columns = ['review', 'label']
    df['label'] = df['label'].map({'truthful': 0, 'deceptive': 1})
    df['review'] = df['review'].str.lower()
    return df

df = load_data()

# ---------------- TRAIN MODEL ----------------
@st.cache_resource
def train_model(df):
    X = df['review']
    y = df['label']

    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=5000
    )

    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    return model, vectorizer, acc

model, vectorizer, accuracy = train_model(df)

# ---------------- RULE-BASED BOOST ----------------
def rule_boost(text):
    score = 0

    if text.count("!") > 2:
        score += 1

    fake_words = ["best", "amazing", "perfect", "guaranteed", "must visit"]
    if any(word in text for word in fake_words):
        score += 1

    words = text.split()
    if len(set(words)) < len(words) * 0.7:
        score += 1

    return score

# ---------------- EXPLAIN ----------------
def explain_review(text):
    reasons = []

    if text.count("!") > 2:
        reasons.append("Too many exclamation marks")

    if any(word in text for word in ["best", "amazing", "perfect"]):
        reasons.append("Overuse of promotional words")

    words = text.split()
    if len(set(words)) < len(words) * 0.7:
        reasons.append("Repetitive wording")

    return ", ".join(reasons) if reasons else "Looks natural"

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Fake Review Detector", page_icon="🧠")

st.title("🧠 Fake Review Detector")
st.write("Check if a review is **Fake or Genuine**")

st.write(f"📊 Model Accuracy: **{accuracy:.2f}**")

user_input = st.text_area("✍️ Enter Review")

if st.button("Check Review"):

    if user_input.strip() == "":
        st.warning("Please enter a review")
    else:
        text = user_input.lower()

        review_vec = vectorizer.transform([text])
        prediction = model.predict(review_vec)
        prob = model.predict_proba(review_vec)
        confidence = max(prob[0]) * 100

        # 🔥 RULE BOOST
        boost = rule_boost(text)
        if boost >= 2:
            prediction[0] = 1

        # RESULT
        if prediction[0] == 1:
            st.error(f"Fake ❌ (Confidence: {confidence:.2f}%)")
        else:
            st.success(f"Genuine ✅ (Confidence: {confidence:.2f}%)")

        # EXPLANATION
        reason = explain_review(text)
        st.info(f"Reason: {reason}")