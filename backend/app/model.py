"""
Model Prediction Module.

Loads the trained model and vectorizer, then provides
prediction functionality with confidence scores and explanations.
"""

import os

import joblib
import numpy as np

from .preprocessing import get_preprocessing_steps, preprocess_text

# Paths to saved model artifacts
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_model")
MODEL_PATH = os.path.join(MODEL_DIR, "classifier.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.joblib")

# Global model instances (loaded once)
_model = None
_vectorizer = None
_metadata = None


def load_model():
    """Load the trained model, vectorizer, and metadata."""
    global _model, _vectorizer, _metadata

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Trained model not found. Please run 'python train_model.py' first."
        )

    _model = joblib.load(MODEL_PATH)
    _vectorizer = joblib.load(VECTORIZER_PATH)
    _metadata = joblib.load(METADATA_PATH)

    return _model, _vectorizer, _metadata


def get_model():
    """Get the loaded model, loading it if necessary."""
    global _model, _vectorizer, _metadata
    if _model is None:
        load_model()
    return _model, _vectorizer, _metadata


def predict(text: str) -> dict:
    """
    Predict whether an internship posting is genuine or fraudulent.

    Args:
        text: Raw text of the internship posting.

    Returns:
        Dictionary with prediction result, confidence score,
        and step-by-step explanation.
    """
    model, vectorizer, metadata = get_model()

    # Get preprocessing steps for explanation
    steps = get_preprocessing_steps(text)

    # Preprocess the text
    processed_text = preprocess_text(text)

    if not processed_text.strip():
        return {
            "prediction": "Unknown",
            "confidence": 0.0,
            "label": -1,
            "explanation": "Could not extract meaningful text for analysis.",
            "steps": steps,
        }

    # Transform text using TF-IDF vectorizer
    text_tfidf = vectorizer.transform([processed_text])

    # Get prediction
    prediction = model.predict(text_tfidf)[0]
    label_name = metadata["labels"][int(prediction)]

    # Get confidence score
    confidence = _get_confidence(model, text_tfidf)

    # Generate explanation
    explanation = _generate_explanation(
        text, processed_text, prediction, confidence, metadata, vectorizer, text_tfidf
    )

    return {
        "prediction": label_name,
        "confidence": round(confidence * 100, 2),
        "label": int(prediction),
        "explanation": explanation,
        "steps": steps,
        "model_used": metadata["model_name"],
        "model_accuracy": round(metadata["accuracy"] * 100, 2),
    }


def _get_confidence(model, text_tfidf) -> float:
    """Extract confidence score from the model prediction."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(text_tfidf)[0]
        return float(max(probabilities))
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(text_tfidf)[0]
        # Convert decision function to pseudo-probability using sigmoid
        confidence = 1 / (1 + np.exp(-abs(float(decision))))
        return confidence
    return 0.75  # Default confidence if method not available


def _generate_explanation(
    original_text: str,
    processed_text: str,
    prediction: int,
    confidence: float,
    metadata: dict,
    vectorizer,
    text_tfidf,
) -> str:
    """Generate a human-readable explanation of the prediction."""
    label = metadata["labels"][int(prediction)]

    # Identify top contributing features
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = text_tfidf.toarray()[0]
    nonzero_indices = tfidf_scores.nonzero()[0]

    top_features = []
    for idx in nonzero_indices:
        top_features.append((feature_names[idx], tfidf_scores[idx]))
    top_features.sort(key=lambda x: x[1], reverse=True)
    top_words = [f for f, s in top_features[:5]]

    # Build explanation
    steps_text = []
    steps_text.append(f"1. **Text Extraction**: Received {len(original_text)} characters of text input.")
    steps_text.append(
        f"2. **NLP Preprocessing**: Cleaned text, removed stopwords, "
        f"and lemmatized to {len(processed_text.split())} key tokens."
    )
    steps_text.append(
        f"3. **Feature Extraction**: Converted text to TF-IDF features "
        f"({metadata['feature_count']} dimensions)."
    )
    steps_text.append(
        f"4. **Classification**: {metadata['model_name']} classifier analyzed the features."
    )
    steps_text.append(
        f"5. **Result**: The posting is classified as **{label}** "
        f"with {confidence*100:.1f}% confidence."
    )

    if top_words:
        steps_text.append(
            f"6. **Key Indicators**: Top contributing words: {', '.join(top_words)}"
        )

    # Add red/green flags
    red_flags = []
    green_flags = []

    text_lower = original_text.lower()
    if any(w in text_lower for w in ["fee", "pay", "payment", "wire", "transfer", "deposit"]):
        red_flags.append("Mentions payment/fees from applicant")
    if any(w in text_lower for w in ["urgent", "act now", "limited", "hurry"]):
        red_flags.append("Uses urgency/pressure tactics")
    if any(w in text_lower for w in ["guaranteed", "100%", "no experience", "no skills"]):
        red_flags.append("Makes unrealistic promises")
    if any(w in text_lower for w in ["ssn", "bank detail", "credit card", "personal id"]):
        red_flags.append("Requests sensitive personal information")
    if any(w in text_lower for w in ["whatsapp", "personal gmail", "telegram"]):
        red_flags.append("Uses informal communication channels")

    if any(w in text_lower for w in ["requirements:", "qualifications", "gpa", "major"]):
        green_flags.append("Lists specific qualifications/requirements")
    if any(w in text_lower for w in ["office location", "campus", "on-site"]):
        green_flags.append("Mentions physical office/location")
    if any(w in text_lower for w in ["apply through", "careers page", "application portal"]):
        green_flags.append("Uses official application channels")
    if any(w in text_lower for w in ["mentorship", "training program", "professional development"]):
        green_flags.append("Offers structured learning/mentorship")

    if red_flags:
        steps_text.append("\n**Red Flags Detected:**")
        for flag in red_flags:
            steps_text.append(f"  - {flag}")

    if green_flags:
        steps_text.append("\n**Green Flags Detected:**")
        for flag in green_flags:
            steps_text.append(f"  - {flag}")

    return "\n".join(steps_text)
