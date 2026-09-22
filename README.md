# Internship Scam Detection Model

An AI-powered application that detects whether an internship opportunity is **Genuine** or **Fraudulent** using **Natural Language Processing (NLP)** and machine learning techniques.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)

## Features

- **Multi-Input Support**: Analyze internship postings via:
  - **Text Description**: Paste internship details directly
  - **URL Link**: Automatically scrapes and analyzes web pages
  - **Image/Poster Upload**: Extracts text using OCR (Tesseract)
- **NLP Pipeline**: Text cleaning, tokenization, stopword removal, lemmatization
- **TF-IDF Feature Extraction**: Converts text to numerical features (up to 5,000 dimensions)
- **ML Classification**: Trained with Naive Bayes, Logistic Regression, and SVM — best model auto-selected
- **Confidence Score**: Shows prediction confidence percentage
- **Step-by-Step Explanation**: Details how the prediction was made, including red/green flags
- **Modern Responsive UI**: Clean dashboard that works on desktop and mobile

## Project Structure

```
Fake-Internship-Detection-/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application & API endpoints
│   │   ├── model.py             # Model loading & prediction logic
│   │   ├── preprocessing.py     # NLP preprocessing pipeline
│   │   ├── scraper.py           # URL text scraping
│   │   └── ocr.py               # Image OCR text extraction
│   ├── train_model.py           # Model training script
│   ├── saved_model/             # Trained model artifacts
│   │   ├── classifier.joblib
│   │   ├── tfidf_vectorizer.joblib
│   │   └── model_metadata.joblib
│   └── data/
│       └── dataset.csv          # Training dataset
├── frontend/
│   ├── index.html               # Main dashboard UI
│   ├── style.css                # Responsive styles
│   └── script.js                # Frontend logic & API integration
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Prerequisites

- **Python 3.9+**
- **Tesseract OCR** (for image text extraction)
- **VS Code** (recommended IDE)

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/jayasri-karunanithi-32/Fake-Internship-Detection-.git
cd Fake-Internship-Detection-
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 5. Install Tesseract OCR

**Windows:**
Download from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH.

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Linux:**
```bash
sudo apt-get install tesseract-ocr
```

### 6. Train the Model

```bash
cd backend
python train_model.py
```

This will:
- Load the dataset (from Kaggle or local CSV)
- Preprocess text using NLP techniques
- Extract TF-IDF features
- Train Naive Bayes, Logistic Regression, and SVM classifiers
- Save the best-performing model

### 7. Run the Backend Server

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

### 8. Open the Frontend

Open `frontend/index.html` in your browser, or serve it with:

```bash
cd frontend
python -m http.server 5500
```

Then visit `http://localhost:5500`.

## Using the Kaggle Dataset

To use the original Kaggle dataset, set up your Kaggle API credentials and update the training script:

```python
import kagglehub
from kagglehub import KaggleDatasetAdapter

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "jayasrikarunanithi/fake-internship-detection",
    ""
)
```

See [Kaggle API docs](https://github.com/Kaggle/kaggle-api#api-credentials) for credential setup.

## API Endpoints

| Method | Endpoint              | Description                        |
|--------|-----------------------|------------------------------------|
| GET    | `/`                   | Health check                       |
| GET    | `/api/health`         | Detailed health status             |
| POST   | `/api/analyze/text`   | Analyze text description           |
| POST   | `/api/analyze/url`    | Analyze URL (scrapes content)      |
| POST   | `/api/analyze/image`  | Analyze uploaded image (OCR)       |

## How It Works

1. **Input**: User submits internship details (text, URL, or image)
2. **Text Extraction**: URL content is scraped; images are processed with OCR
3. **NLP Preprocessing**:
   - Text cleaning (remove URLs, emails, special characters)
   - Tokenization (split into words)
   - Stopword removal (remove common words)
   - Lemmatization (reduce words to base form)
4. **Feature Extraction**: TF-IDF vectorization (unigrams + bigrams)
5. **Classification**: Trained ML model predicts Genuine vs. Fraudulent
6. **Output**: Prediction result, confidence score, and detailed explanation with red/green flags

## Technologies Used

- **Backend**: Python, FastAPI, Uvicorn
- **ML/NLP**: scikit-learn, NLTK, TF-IDF
- **OCR**: Tesseract, Pillow
- **Web Scraping**: BeautifulSoup, Requests
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Dataset**: Kaggle (`jayasrikarunanithi/fake-internship-detection`)

## License

This project is developed for academic purposes as a final-year project.
