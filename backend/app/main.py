"""
FastAPI Backend for Internship Scam Detection.

Provides REST API endpoints for analyzing internship postings
via text, URL, or image upload.
"""

import os
import sys

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.model import predict
from app.ocr import extract_text_from_image
from app.scraper import scrape_url

app = FastAPI(
    title="Internship Scam Detection API",
    description="Detects whether an internship opportunity is Genuine or Fraudulent using NLP and ML.",
    version="1.0.0",
)

# CORS middleware - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")
if os.path.exists(frontend_dir):
    pass
    

@app.get("/")
async def root():
    return FileResponse(os.path.join(frontend_dir, "index.html"))


@app.get("/api/health")
async def health_check():
    """Detailed health check."""
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_model")
    model_exists = os.path.exists(os.path.join(model_dir, "classifier.joblib"))
    return {
        "status": "healthy",
        "model_loaded": model_exists,
        "api_version": "1.0.0",
    }


@app.post("/api/analyze/text")
async def analyze_text(text: str = Form(...)):
    """
    Analyze internship description text for fraud detection.

    Args:
        text: The internship description text to analyze.
    """
    if not text or not text.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "Please provide non-empty text to analyze."},
        )

    try:
        result = predict(text)
        return {"success": True, "input_type": "text", "result": result}
    except FileNotFoundError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Analysis failed: {str(e)}"},
        )


@app.post("/api/analyze/url")
async def analyze_url(url: str = Form(...)):
    """
    Analyze an internship posting URL for fraud detection.

    Args:
        url: The URL of the internship posting to analyze.
    """
    if not url or not url.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "Please provide a valid URL."},
        )

    try:
        # Step 1: Scrape text from URL
        scraped_text = scrape_url(url)

        # Step 2: Run prediction
        result = predict(scraped_text)
        result["scraped_text_preview"] = scraped_text[:500] + (
            "..." if len(scraped_text) > 500 else ""
        )
        return {"success": True, "input_type": "url", "result": result}
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except FileNotFoundError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"URL analysis failed: {str(e)}"},
        )


@app.post("/api/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze an internship poster/image for fraud detection using OCR.

    Args:
        file: The image file to analyze.
    """
    if not file:
        return JSONResponse(
            status_code=400,
            content={"error": "Please upload an image file."},
        )

    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/gif", "image/bmp", "image/webp", "image/tiff"]
    if file.content_type and file.content_type not in allowed_types:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported file type: {file.content_type}. Please upload an image file."},
        )

    try:
        # Step 1: Read image bytes
        image_bytes = await file.read()

        # Step 2: Extract text using OCR
        extracted_text = extract_text_from_image(image_bytes)

        # Step 3: Run prediction
        result = predict(extracted_text)
        result["extracted_text_preview"] = extracted_text[:500] + (
            "..." if len(extracted_text) > 500 else ""
        )
        return {"success": True, "input_type": "image", "result": result}
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except FileNotFoundError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Image analysis failed: {str(e)}"},
        )

app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
