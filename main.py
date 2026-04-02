import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack
import tkinter as tk
from tkinter import messagebox

# -------- LOAD DATA --------
df = pd.read_csv("data/deceptive-opinion.csv")

# Keep required columns
df = df[['text', 'deceptive', 'polarity']]
df.columns = ['review', 'label', 'polarity']

# Convert labels
df['label'] = df['label'].map({'truthful': 0, 'deceptive': 1})

# Lowercase
df['review'] = df['review'].str.lower()

# -------- FEATURE ENGINEERING --------

# Text feature
X_text = df['review']

# Polarity feature
X_extra = df['polarity'].map({'positive': 1, 'negative': 0}).values.reshape(-1,1)

# Keyword feature
fake_words = ['amazing', 'perfect', 'best', 'must', 'recommended']
df['fake_score'] = df['review'].apply(
    lambda x: sum(word in x for word in fake_words)
)
X_extra2 = df['fake_score'].values.reshape(-1,1)

# Target
y = df['label']

# -------- VECTORIZATION --------
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1,2),
    token_pattern=r'(?u)\b\w+\b|!+'
)

X_text_vec = vectorizer.fit_transform(X_text)

# Combine features
X_combined = hstack([X_text_vec, X_extra, X_extra2])

# -------- TRAIN TEST SPLIT --------
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2
)

# -------- MODEL --------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------- EVALUATION --------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("\nModel trained successfully\n")

# -------- EXPLANATION FUNCTION --------
def explain_review(text):
    reasons = []

    words = text.split()

    # Repetition
    if len(words) != len(set(words)):
        reasons.append("Repeated words")

    # Exclamation
    if "!" in text:
        reasons.append("Excessive punctuation")

    # Promotional words
    strong_words = ['amazing', 'perfect', 'best', 'must', 'recommended']
    if any(word in text for word in strong_words):
        reasons.append("Promotional language")

    # Short review
    if len(words) < 5:
        reasons.append("Very short review")

    if not reasons:
        return "Looks natural"

    return ", ".join(reasons)

# -------- GUI FUNCTION --------
def predict_review():
    user_input = entry.get().strip().lower()

    if user_input == "":
        messagebox.showwarning("Warning", "Please enter a review")
        return

    # Vectorize
    review_vec = vectorizer.transform([user_input])

    # Default polarity
    extra_feature = np.array([[1]])

    # Keyword score
    fake_score = sum(word in user_input for word in fake_words)
    extra_feature2 = np.array([[fake_score]])

    # Combine
    final_input = hstack([review_vec, extra_feature, extra_feature2])

    prediction = model.predict(final_input)
    prob = model.predict_proba(final_input)

    confidence = max(prob[0]) * 100

    # Prediction result
    if prediction[0] == 1:
        result_label.config(text="Fake ❌", fg="red")
    else:
        result_label.config(text="Genuine ✅", fg="green")

    confidence_label.config(text=f"Confidence: {confidence:.2f}%")

    # Explanation
    reason = explain_review(user_input)
    reason_label.config(text=f"Reason: {reason}")

# -------- GUI WINDOW --------
root = tk.Tk()
root.title("Fake Review Detector")
root.geometry("500x400")
root.configure(bg="lightblue")

# Title
title = tk.Label(root, text="Fake Review Detector",
                 font=("Arial", 18, "bold"), bg="lightblue")
title.pack(pady=10)

# Input box
entry = tk.Entry(root, width=55, font=("Arial", 12))
entry.pack(pady=10)

# Button
predict_btn = tk.Button(root, text="Check Review",
                        command=predict_review,
                        bg="blue", fg="white")
predict_btn.pack(pady=10)

# Result
result_label = tk.Label(root, text="", font=("Arial", 16), bg="lightblue")
result_label.pack(pady=10)

# Confidence
confidence_label = tk.Label(root, text="", font=("Arial", 12), bg="lightblue")
confidence_label.pack(pady=5)

# Reason
reason_label = tk.Label(root, text="", font=("Arial", 10),
                        bg="lightblue", wraplength=450)
reason_label.pack(pady=5)

# Run app
root.mainloop()