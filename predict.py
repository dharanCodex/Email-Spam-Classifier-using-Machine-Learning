"""
Email Spam Classifier - CLI Prediction Tool
============================================
Load the trained model and vectorizer, then classify user-entered emails.
"""

import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)

# Load saved model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    """Clean and normalize text (same as training pipeline)."""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)


def predict_email(email_text):
    """Predict whether an email is spam or not."""
    cleaned = preprocess_text(email_text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    confidence = max(probability) * 100
    label = "🚫 Spam Email" if prediction == 1 else "✅ Not Spam Email"
    return label, confidence


if __name__ == "__main__":
    print("=" * 50)
    print("  EMAIL SPAM CLASSIFIER - Prediction")
    print("=" * 50)
    print("Type 'quit' to exit.\n")

    while True:
        email = input("📧 Enter email text: ").strip()
        if email.lower() == "quit":
            print("\nGoodbye! 👋")
            break
        if not email:
            print("⚠️  Please enter some text.\n")
            continue

        label, confidence = predict_email(email)
        print(f"\n   Result: {label}")
        print(f"   Confidence: {confidence:.1f}%\n")
