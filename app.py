import re
import joblib
import gradio as gr

# Load trained model and TF-IDF vectorizer
model = joblib.load("spam_email_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"https?://|www\.|https?S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text):
    stop_words = {
        "i", "me", "my", "myself", "we", "our", "ours", "you", "your",
        "he", "him", "his", "she", "her", "it", "its", "they", "them",
        "this", "that", "these", "those", "am", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
        "while", "of", "at", "by", "for", "with", "about", "against",
        "between", "into", "through", "during", "before", "after", "above",
        "below", "to", "from", "in", "out", "on", "off", "over", "under",
        "again", "further", "then", "once", "here", "there", "when",
        "where", "why", "how", "all", "any", "both", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "can", "will", "just", "should",
        "now"
    }

    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


def predict_spam(message):
    if not message or not message.strip():
        return "Please enter an email or message."

    cleaned = clean_text(message)
    processed = remove_stopwords(cleaned)

    features = vectorizer.transform([processed])
    prediction = model.predict(features)[0]

    if prediction == 1:
        return "🚨 SPAM EMAIL DETECTED"
    else:
        return "✅ NOT SPAM — This appears to be a normal message"


demo = gr.Interface(
    fn=predict_spam,
    inputs=gr.Textbox(
        lines=6,
        placeholder="Paste your email or message here..."
    ),
    outputs=gr.Textbox(label="Detection Result"),
    title="📧 Spam Email Detection System",
    description="Enter an email or message to check whether it is Spam or Not Spam.",
    submit_btn="Check Email",
    clear_btn="Clear"
)


if __name__ == "__main__":
    demo.launch()