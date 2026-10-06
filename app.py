"""
Email Spam Classifier - Flask Web Application
==============================================
Serves the modern dashboard UI with prediction API,
metrics endpoint, and PDF report download.
"""

import os
import json
import re
import io
import sys
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from flask import Flask, render_template, request, jsonify, send_file
from datetime import datetime

nltk.download("stopwords", quiet=True)

app = Flask(__name__)

# ── Load Model & Vectorizer ──────────────────────────────────────────────────
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


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the dashboard page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Predict whether an email is spam or not spam."""
    data = request.get_json()
    email_text = data.get("email", "").strip()

    if not email_text:
        return jsonify({"error": "No email text provided"}), 400

    cleaned = preprocess_text(email_text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    confidence = round(max(probability) * 100, 1)

    result = "Spam Email" if prediction == 1 else "Not Spam Email"
    is_spam = bool(prediction == 1)

    return jsonify({
        "result": result,
        "is_spam": is_spam,
        "confidence": confidence,
    })


@app.route("/metrics")
def metrics():
    """Return saved model metrics for the dashboard charts."""
    try:
        with open("metrics.json", "r") as f:
            data = json.load(f)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({"error": "metrics.json not found. Run train.py first."}), 404


@app.route("/download-report")
def download_report():
    """Generate and return a PDF report with model metrics."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import inch, cm
        from reportlab.lib.colors import HexColor
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        return jsonify({"error": "reportlab not installed. Run: pip install reportlab"}), 500

    # Load metrics
    try:
        with open("metrics.json", "r") as f:
            metrics_data = json.load(f)
    except FileNotFoundError:
        return jsonify({"error": "metrics.json not found. Run train.py first."}), 404

    # Create PDF in memory
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=22,
        textColor=HexColor("#0a192f"),
        spaceAfter=20,
        alignment=TA_CENTER,
    )
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=HexColor("#1a73e8"),
        spaceBefore=20,
        spaceAfter=10,
    )
    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["Normal"],
        fontSize=11,
        textColor=HexColor("#333333"),
        spaceAfter=6,
    )

    elements = []

    # Title
    elements.append(Paragraph("Email Spam Classifier — Model Report", title_style))
    elements.append(Paragraph(
        f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        ParagraphStyle("Date", parent=body_style, alignment=TA_CENTER, textColor=HexColor("#666666"))
    ))
    elements.append(Spacer(1, 20))

    # Project Overview
    elements.append(Paragraph("Project Overview", heading_style))
    elements.append(Paragraph(
        "This report summarizes the performance of a Multinomial Naive Bayes "
        "classifier trained to detect spam emails using TF-IDF vectorization.",
        body_style,
    ))
    elements.append(Spacer(1, 10))

    # Dataset Info
    elements.append(Paragraph("Dataset Summary", heading_style))
    dataset_data = [
        ["Metric", "Value"],
        ["Total Samples", str(metrics_data.get("total_samples", "N/A"))],
        ["Training Samples", str(metrics_data.get("train_samples", "N/A"))],
        ["Test Samples", str(metrics_data.get("test_samples", "N/A"))],
        ["Spam Messages", str(metrics_data.get("spam_count", "N/A"))],
        ["Ham Messages", str(metrics_data.get("ham_count", "N/A"))],
    ]
    table = Table(dataset_data, colWidths=[3 * inch, 2.5 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a73e8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f8f9fa"), HexColor("#ffffff")]),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 15))

    # Performance Metrics
    elements.append(Paragraph("Model Performance", heading_style))
    perf_data = [
        ["Metric", "Score"],
        ["Accuracy", f"{metrics_data.get('accuracy', 0):.2f}%"],
        ["Precision", f"{metrics_data.get('precision', 0):.2f}%"],
        ["Recall", f"{metrics_data.get('recall', 0):.2f}%"],
        ["F1 Score", f"{metrics_data.get('f1_score', 0):.2f}%"],
    ]
    table2 = Table(perf_data, colWidths=[3 * inch, 2.5 * inch])
    table2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#0a192f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#64ffda")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f0f7ff"), HexColor("#ffffff")]),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(table2)
    elements.append(Spacer(1, 15))

    # Confusion Matrix Image
    cm_path = os.path.join("static", "confusion_matrix.png")
    if os.path.exists(cm_path):
        elements.append(Paragraph("Confusion Matrix", heading_style))
        img = Image(cm_path, width=4 * inch, height=3.3 * inch)
        elements.append(img)
        elements.append(Spacer(1, 15))

    # Workflow
    elements.append(Paragraph("ML Workflow", heading_style))
    workflow_steps = [
        "1. Dataset (spam.csv)",
        "2. Data Preprocessing (lowercase, remove punctuation, stopwords, stemming)",
        "3. TF-IDF Vectorization (max_features=5000)",
        "4. Train-Test Split (80/20)",
        "5. Multinomial Naive Bayes Training",
        "6. Model Evaluation (Accuracy, Precision, Recall, F1)",
        "7. Save Model (model.pkl, vectorizer.pkl)",
        "8. Flask Web Application",
        "9. Real-time Prediction",
    ]
    for step in workflow_steps:
        elements.append(Paragraph(f"  {step}", body_style))

    elements.append(Spacer(1, 20))
    elements.append(Paragraph(
        "— Generated by Email Spam Classifier Dashboard —",
        ParagraphStyle("Footer", parent=body_style, alignment=TA_CENTER, textColor=HexColor("#999999"), fontSize=9),
    ))

    doc.build(elements)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"spam_classifier_report_{datetime.now().strftime('%Y%m%d')}.pdf",
        mimetype="application/pdf",
    )


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n" + "=" * 50)
    print("  [WEB] Email Spam Classifier Dashboard")
    print("  [URL] http://127.0.0.1:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, port=5000)
