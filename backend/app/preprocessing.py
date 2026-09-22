"""
NLP Preprocessing Module for Internship Scam Detection.

Handles text cleaning, tokenization, stopword removal,
lemmatization, and feature extraction using TF-IDF.
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure NLTK data is available
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Remove URLs, emails, special characters, and extra whitespace."""
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def tokenize_text(text: str) -> list[str]:
    """Tokenize text into individual words."""
    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove common English stopwords from token list."""
    return [token for token in tokens if token not in stop_words and len(token) > 2]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Lemmatize tokens to their base form."""
    return [lemmatizer.lemmatize(token) for token in tokens]


def preprocess_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline:
    1. Text cleaning
    2. Tokenization
    3. Stopword removal
    4. Lemmatization
    5. Rejoin into clean string
    """
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)


def get_preprocessing_steps(text: str) -> dict:
    """Return intermediate results of each preprocessing step for explanation."""
    original = text
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    after_stopwords = remove_stopwords(tokens)
    lemmatized = lemmatize_tokens(after_stopwords)
    final = " ".join(lemmatized)

    return {
        "original": original[:200] + ("..." if len(original) > 200 else ""),
        "cleaned": cleaned[:200] + ("..." if len(cleaned) > 200 else ""),
        "tokens_count": len(tokens),
        "after_stopword_removal": len(after_stopwords),
        "final_processed": final[:200] + ("..." if len(final) > 200 else ""),
    }
