# 📧 Spam Email Detection System

A Machine Learning-based web application that detects whether an email or text message is **Spam** or **Not Spam**.

## 🚀 Features

- 📩 Enter or paste an email/message
- 🤖 Machine Learning-based spam detection
- 🔤 TF-IDF text vectorization
- ⚡ Fast prediction
- 🌐 Interactive Gradio web interface
- ✅ Clear Spam / Not Spam prediction

## 🛠️ Technologies Used

- Python
- Scikit-learn
- TF-IDF Vectorization
- Gradio
- Pandas
- Jupyter Notebook

## 🧠 Machine Learning

The system uses a Machine Learning classification model trained on email/text data.

The text is processed using **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization before being passed to the trained model for prediction.

### Prediction Classes

- 🟢 **Not Spam**
- 🔴 **Spam**

## 📂 Project Structure

```text
Spam-Email-Detection-ML/
│
├── app.py
├── Spam_Email_Detection_ML.ipynb
├── requirements.txt
├── spam_email_model.pkl
├── tfidf_vectorizer.pkl
├── README.md
└── .gitignore
