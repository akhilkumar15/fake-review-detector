# 🧠 AI Fake Review Detector

## About the Project

We see online reviews everywhere—hotels, restaurants, products, and services. But not every review is genuine. Some are written just to promote a product or influence customer opinions.

To explore this problem, I built an AI-based Fake Review Detector using Machine Learning and Natural Language Processing (NLP). The application analyzes review text and predicts whether it is likely to be genuine or deceptive.

The project is deployed as a live web application, allowing users to test reviews in real time.

---

## 🌐 Live Demo

Try the deployed application:

[https://fake-review-detector-2a7ma8ioonwhqrpznqlpns.streamlit.app/](https://fake-review-detector-2a7ma8ioonwhqrpznqlpns.streamlit.app/)

---

## What the Application Does

* Accepts review text from the user
* Predicts whether the review is Fake or Genuine
* Displays a confidence score for the prediction
* Uses additional rule-based checks to identify suspicious writing patterns
* Provides instant results through a simple web interface

---

## How It Works

The workflow of the project is:

```text
Review Input
      ↓
Text Preprocessing
      ↓
TF-IDF Vectorization
      ↓
Logistic Regression Model
      ↓
Rule-Based Analysis
      ↓
Fake / Genuine Prediction
```

## Technologies Used

* Python
* Streamlit
* Scikit-Learn
* TF-IDF Vectorizer
* Logistic Regression
* Pandas
* NumPy

---

## Dataset

The model was trained using the Deceptive Opinion Spam Dataset.

Dataset Statistics:

* Total Reviews: 1600
* Genuine Reviews: 800
* Fake Reviews: 800

The balanced dataset helped train the model without favoring one class over another.

---

## Model Performance

The model achieves approximately 85% accuracy on the test dataset.

To evaluate performance, I used:

* Accuracy Score
* Precision
* Recall
* F1 Score

In addition to machine learning predictions, I added simple rule-based checks to identify patterns commonly found in fake reviews, such as excessive promotional language and repeated words.

---

## Screenshots

### Home Page

<img width="1919" height="1031" alt="Screenshot 2026-06-02 224950" src="https://github.com/user-attachments/assets/58983149-800e-4fd9-90f7-21f6817c625b" />



### Fake Review Detection

<img width="1919" height="1031" alt="Screenshot 2026-06-02 225255" src="https://github.com/user-attachments/assets/5dc8a886-9918-4ce0-b055-8bc04b203bf9" />




### Genuine Review Detection

<img width="1919" height="1025" alt="Screenshot 2026-06-02 225323" src="https://github.com/user-attachments/assets/c618fe6c-3db6-4a5a-84dd-cae43ee0d012" />


---

## Challenges I Faced

While building this project, I faced a few challenges:

* Understanding how to convert text into numerical features
* Improving predictions for highly promotional reviews
* Deploying the application online using Streamlit Cloud
* Handling cases where genuine reviews looked similar to fake reviews

Working through these issues helped me understand practical NLP workflows and machine learning deployment.

---

## Future Improvements

Some improvements I would like to add in future versions:

* BERT-based deep learning model
* Better explainability for predictions
* Sentiment analysis integration
* Larger and more diverse datasets
* Improved user interface

---

## Source Code

GitHub Repository:
https://github.com/akhilkumar15/fake-review-detector

## Author

Akhil Kumar

This project helped me gain practical experience in Machine Learning, Natural Language Processing (NLP), model evaluation, and deployment using Streamlit.
