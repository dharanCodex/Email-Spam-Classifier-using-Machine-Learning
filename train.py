"""
Email Spam Classifier - Training Pipeline
==========================================
Dataset → Preprocessing → TF-IDF Vectorization → Train-Test Split
→ Multinomial Naive Bayes → Evaluation → Save Model
"""

import pandas as pd
import numpy as np
import re
import json
import os
import sys
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── 1. Download NLTK data ────────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── 2. Load Dataset ──────────────────────────────────────────────────────────
print("=" * 60)
print("  EMAIL SPAM CLASSIFIER - TRAINING PIPELINE")
print("=" * 60)

dataset_path = os.path.join("dataset", "spam.csv")
print(f"\n[*] Loading dataset from: {dataset_path}")

try:
    df = pd.read_csv(dataset_path, encoding="latin-1")
except UnicodeDecodeError:
    df = pd.read_csv(dataset_path, encoding="utf-8")

# Drop unnamed columns (common in Kaggle version)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Rename columns
df.columns = ["label", "message"] if len(df.columns) == 2 else df.columns
print(f"   Dataset loaded: {len(df)} rows")
print(f"   Spam: {len(df[df['label'] == 'spam'])} | Ham: {len(df[df['label'] == 'ham'])}")

# ── 3. Preprocessing ─────────────────────────────────────────────────────────
print("\n[*] Preprocessing text data...")

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    """Clean and normalize text for classification."""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)          # Remove non-alphabetic chars
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)


df["clean_message"] = df["message"].apply(preprocess_text)

# Encode labels: spam = 1, ham = 0
df["label_num"] = df["label"].map({"spam": 1, "ham": 0})

print("   [OK] Text cleaned, stopwords removed, stemming applied")

# ── 4. TF-IDF Vectorization ──────────────────────────────────────────────────
print("\n[*] Applying TF-IDF Vectorization (max_features=5000)...")

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["clean_message"])
y = df["label_num"]

print(f"   Feature matrix shape: {X.shape}")

# ── 5. Train-Test Split ──────────────────────────────────────────────────────
print("\n[*] Splitting data: 80% Train / 20% Test...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

# ── 6. Train Naive Bayes Model ───────────────────────────────────────────────
print("\n[*] Training Multinomial Naive Bayes classifier...")

model = MultinomialNB()
model.fit(X_train, y_train)

print("   [OK] Model trained successfully!")

# ── 7. Evaluation ────────────────────────────────────────────────────────────
print("\n[*] Evaluating model performance...")
print("-" * 40)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print(f"   Accuracy  : {accuracy * 100:.2f}%")
print(f"   Precision : {precision * 100:.2f}%")
print(f"   Recall    : {recall * 100:.2f}%")
print(f"   F1 Score  : {f1 * 100:.2f}%")
print()
print("   Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
print("   Confusion Matrix:")
print(f"   {cm}")

# ── 8. Generate Confusion Matrix Heatmap ─────────────────────────────────────
print("\n[*] Generating confusion matrix heatmap...")

os.makedirs("static", exist_ok=True)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"],
    linewidths=1,
    linecolor="#1e3a5f",
    cbar_kws={"shrink": 0.8},
    annot_kws={"size": 16, "fontweight": "bold"},
    ax=ax,
)
ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
ax.set_ylabel("Actual", fontsize=12, fontweight="bold")
ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
fig.patch.set_facecolor("#0a192f")
ax.set_facecolor("#0a192f")
ax.title.set_color("white")
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
ax.tick_params(colors="white")
plt.tight_layout()
plt.savefig(
    os.path.join("static", "confusion_matrix.png"),
    dpi=150,
    bbox_inches="tight",
    facecolor="#0a192f",
)
plt.close()
print("   [OK] Saved: static/confusion_matrix.png")

# ── 9. Save Model, Vectorizer & Metrics ──────────────────────────────────────
print("\n[*] Saving model artifacts...")

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

metrics = {
    "accuracy": round(accuracy * 100, 2),
    "precision": round(precision * 100, 2),
    "recall": round(recall * 100, 2),
    "f1_score": round(f1 * 100, 2),
    "confusion_matrix": cm.tolist(),
    "total_samples": len(df),
    "train_samples": X_train.shape[0],
    "test_samples": X_test.shape[0],
    "spam_count": int(df["label_num"].sum()),
    "ham_count": int((df["label_num"] == 0).sum()),
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("   [OK] model.pkl")
print("   [OK] vectorizer.pkl")
print("   [OK] metrics.json")

print("\n" + "=" * 60)
print("  TRAINING COMPLETE -- ALL ARTIFACTS SAVED!")
print("=" * 60)
print("\nNext steps:")
print("  1. Run 'python predict.py' to test from command line")
print("  2. Run 'python app.py' to launch the web dashboard")
print()
