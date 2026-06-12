import streamlit as st
import joblib

# Page config
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# Load model
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Title
st.title("📰 Fake News Detection System")
st.markdown("""
Detect whether a news article is **Real** or **Fake**
using **TF-IDF + Logistic Regression Machine Learning Model**.
""")

st.info("Model Accuracy: 98.46%")

# Input
news = st.text_area(
    "Paste News Article Here",
    height=250,
    placeholder="Enter news content..."
)

# Predict
if st.button("🔍 Analyze News", use_container_width=True):

    if news.strip() == "":
        st.warning("Please enter some news text.")
    else:
        transformed = vectorizer.transform([news])

        prediction = model.predict(transformed)

        if prediction[0] == 0:
            st.error("🚨 Prediction: FAKE NEWS")
        else:
            st.success("✅ Prediction: REAL NEWS")

# Sidebar
st.sidebar.title("Project Information")

st.sidebar.markdown("""
### Technologies Used
- Python
- Pandas
- NumPy
- Scikit-Learn
- TF-IDF Vectorizer
- Logistic Regression
- Streamlit

### Dataset
Real and Fake News Dataset

### Accuracy
98.46%
""")

st.markdown("---")
st.caption("Built using Machine Learning and Natural Language Processing")