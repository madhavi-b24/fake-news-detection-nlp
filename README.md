# 📰 Fake News Detection using Logistic Regression

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Logistic%20Regression-green)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-orange)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)

## 📌 Overview

This project is a Machine Learning based Fake News Detection System that classifies news articles as **Real** or **Fake** using Natural Language Processing (NLP) techniques.

The model is trained on thousands of real and fake news articles, where textual content is transformed into numerical features using **TF-IDF Vectorization** and classified using **Logistic Regression**.

A Streamlit web application is integrated for real-time predictions.

---

## 🚀 Features

✅ News Classification (Real/Fake)

✅ NLP Text Preprocessing

✅ TF-IDF Feature Extraction

✅ Logistic Regression Model

✅ Interactive Streamlit Web App

✅ Model Persistence using Pickle/Joblib

✅ Real-Time News Prediction

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Libraries
- Pandas
- NumPy
- Scikit-Learn
- Joblib
- Streamlit

### Machine Learning
- Logistic Regression

### NLP Techniques
- Text Cleaning
- Lowercasing
- Punctuation Removal
- TF-IDF Vectorization

---

## 📂 Project Structure

```text
fake-news-detection-nlp/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── src/
│   └── train.py
│
├── models/
│   ├── fake_news_model.pkl
│   └── vectorizer.pkl
```

---

## 🔄 Machine Learning Workflow

### 1. Data Collection

Dataset consists of:

- Fake News Articles
- Real News Articles

### 2. Data Preprocessing

- Remove special characters
- Convert text to lowercase
- Remove unwanted symbols
- Clean textual data

### 3. Feature Extraction

TF-IDF Vectorization converts text into numerical features suitable for Machine Learning algorithms.

### 4. Model Training

Algorithm Used:

```text
Logistic Regression
```

### 5. Evaluation

The model is evaluated on unseen test data.

---

## 📊 Results

### Accuracy Achieved

```text
98.46%
```

The Logistic Regression model demonstrated strong performance in distinguishing between real and fake news articles.

---

## 🖥️ Streamlit Application

The project includes a user-friendly Streamlit interface.

Users can:

1. Enter a news article
2. Click Analyze News
3. Receive prediction instantly

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detection-nlp.git
```

Move into project folder:

```bash
cd fake-news-detection-nlp
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Train Model

```bash
python src/train.py
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

---

## 📈 Future Improvements

- Deep Learning Models (LSTM, GRU)
- Transformer Models (BERT)
- News Source Verification
- Confidence Score Visualization
- Cloud Deployment

---

## 👨‍💻 Author

**Madhavi Bonda**

B.Tech Information Technology

Machine Learning & AI Enthusiast

---

## ⭐ If you found this project useful, consider giving it a star.