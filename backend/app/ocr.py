"""
OCR Module for Image Text Extraction.

Uses Tesseract OCR to extract text from uploaded
internship poster images.
"""

from email.mime import image
import io
from unittest import result

from rapidocr_onnxruntime import RapidOCR
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

    ocr = RapidOCR()
    result, _ = ocr(image)
    text = "\n".join([item[1] for item in result]) if result else ""

    if not text.strip():
        raise ValueError(
            "No text could be extracted from the image. "
            "Please ensure the image contains readable text."
        )

    return text.strip()
