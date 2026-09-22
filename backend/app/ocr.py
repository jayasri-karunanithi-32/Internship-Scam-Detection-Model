"""
OCR Module for Image Text Extraction.

Uses Tesseract OCR to extract text from uploaded
internship poster images.
"""

import io

import pytesseract
from PIL import Image


def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Extract text from an image using Tesseract OCR.

    Args:
        image_bytes: Raw bytes of the uploaded image.

    Returns:
        Extracted text from the image.

    Raises:
        ValueError: If no text could be extracted.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise ValueError(f"Invalid image file: {e}") from e

    # Convert to RGB if necessary (handles RGBA, palette, etc.)
    if image.mode not in ("L", "RGB"):
        image = image.convert("RGB")

    text = pytesseract.image_to_string(image)

    if not text.strip():
        raise ValueError(
            "No text could be extracted from the image. "
            "Please ensure the image contains readable text."
        )

    return text.strip()
