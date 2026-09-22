"""
Model Training Script for Internship Scam Detection.

Loads dataset (from Kaggle or local CSV), preprocesses text,
extracts TF-IDF features, trains multiple classifiers,
selects the best one, and saves it.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.preprocessing import preprocess_text


def load_dataset() -> pd.DataFrame:
    """Load dataset from Kaggle or fall back to local CSV."""
    # Try loading from Kaggle first
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter

        print("Attempting to load dataset from Kaggle...")
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "jayasrikarunanithi/fake-internship-detection",
            sql_query="SELECT * FROM fake-internship-detection",
        )
        print(f"Loaded {len(df)} records from Kaggle.")
        return df
    except Exception as e:
        print(f"Kaggle loading failed ({e}), using local dataset...")

    # Fall back to local CSV
    local_path = os.path.join(os.path.dirname(__file__), "data", "dataset.csv")
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        print(f"Loaded {len(df)} records from local CSV.")
        return df

    # Generate synthetic dataset if nothing else is available
    print("No dataset found. Generating synthetic training data...")
    return generate_synthetic_dataset()


def generate_synthetic_dataset() -> pd.DataFrame:
    """Generate a synthetic dataset for internship scam detection."""
    genuine_examples = [
        "Software Engineering Intern at Google. Requirements: Currently pursuing BS/MS in Computer Science. Strong programming skills in Python or Java. Duration: 12 weeks. Paid internship with competitive stipend. Apply through our careers page.",
        "Marketing Intern at Deloitte. Join our marketing team for a 6-month internship. Requirements: Marketing or Business major, strong communication skills. Monthly stipend provided. Office location: New York.",
        "Data Science Intern at Microsoft Research. Work on cutting-edge AI projects. Requirements: PhD student in ML/AI, published research preferred. 3-month paid internship. Apply with CV and research statement.",
        "Finance Intern at JPMorgan Chase. Rotational program across investment banking divisions. Requirements: Finance or Economics major, GPA 3.5+. 10-week summer program with housing allowance.",
        "UX Design Intern at Apple. Help design next-generation products. Requirements: Design portfolio, proficiency in Figma/Sketch. 12-week paid internship at Cupertino campus.",
        "Research Intern at IBM Watson. Work on natural language processing research. Requirements: Graduate student in CS/NLP. Published papers preferred. Competitive stipend and relocation assistance.",
        "Product Management Intern at Amazon. Drive product strategy for AWS services. Requirements: MBA candidate, analytical mindset. 12-week paid internship in Seattle.",
        "Civil Engineering Intern at AECOM. Support infrastructure projects. Requirements: Civil Engineering student, AutoCAD proficiency. 3-month internship with mentorship program.",
        "Human Resources Intern at PwC. Support talent acquisition initiatives. Requirements: HR or Psychology major. 6-month internship with professional development opportunities.",
        "Cybersecurity Intern at Cisco. Work on network security projects. Requirements: CS or Information Security student. CompTIA Security+ preferred. Paid summer internship.",
        "Mechanical Engineering Intern at Tesla. Support vehicle design and manufacturing. Requirements: ME student with CAD experience. 3-month paid internship in Fremont, CA.",
        "Content Writing Intern at HubSpot. Create blog posts and marketing content. Requirements: English or Journalism major, writing samples required. Remote internship with monthly stipend.",
        "Accounting Intern at Ernst & Young. Support audit and assurance services. Requirements: Accounting major, working toward CPA. 8-week summer internship.",
        "Biotech Research Intern at Genentech. Lab-based research in drug development. Requirements: Biology or Chemistry graduate student. 12-week paid internship.",
        "Cloud Engineering Intern at Salesforce. Build and deploy cloud solutions. Requirements: CS student with AWS/Azure experience. Paid internship with full-time conversion opportunity.",
        "Legal Intern at Baker McKenzie. Support corporate law practice. Requirements: Law student (2L or 3L), law review preferred. Paid summer associate position.",
        "Supply Chain Intern at Procter & Gamble. Optimize logistics and distribution. Requirements: Supply Chain or Operations major. 10-week paid internship with relocation support.",
        "Graphic Design Intern at Adobe. Create visual content for marketing campaigns. Requirements: Design student with Adobe Creative Suite proficiency. Portfolio required. Remote option available.",
        "Environmental Science Intern at EPA. Support environmental research and policy analysis. Requirements: Environmental Science major, GIS skills preferred. Government stipend provided.",
        "Machine Learning Intern at NVIDIA. Develop deep learning models for GPU optimization. Requirements: MS/PhD in CS with ML focus. Strong PyTorch/TensorFlow skills. Competitive compensation.",
        "Business Analyst Intern at McKinsey. Support consulting engagements across industries. Requirements: Top-tier university, analytical skills. 10-week summer program.",
        "Pharmaceutical Intern at Pfizer. Support clinical trial data analysis. Requirements: Pharmacy or Life Sciences student. 12-week paid internship.",
        "Robotics Intern at Boston Dynamics. Work on robot locomotion and control. Requirements: Robotics or ME graduate student. C++ and ROS experience required.",
        "Social Media Intern at Meta. Manage content strategy for internal communications. Requirements: Communications major, social media analytics experience. Paid internship at Menlo Park.",
        "Quality Assurance Intern at Intel. Test semiconductor manufacturing processes. Requirements: EE or Materials Science student. Clean room experience a plus. Paid internship.",
        "Database Administration Intern at Oracle. Support enterprise database solutions. Requirements: CS student with SQL proficiency. 3-month paid internship with training program.",
        "Architecture Intern at Gensler. Support commercial building design projects. Requirements: Architecture student, Revit proficiency. Paid internship with design studio access.",
        "Journalism Intern at The New York Times. Report on technology and business stories. Requirements: Journalism student, published clips required. Paid 10-week summer internship.",
        "Aerospace Engineering Intern at SpaceX. Support rocket propulsion systems. Requirements: AE student with thermal analysis skills. On-site at Hawthorne, CA. Paid internship.",
        "Network Engineering Intern at Juniper Networks. Configure and test network equipment. Requirements: CS or IT student, CCNA preferred. 12-week paid internship.",
    ]

    fraudulent_examples = [
        "URGENT! Work from home internship. Earn $5000/week with just 2 hours of work daily! No experience needed. Send $200 registration fee to secure your spot. Limited positions available!",
        "Congratulations! You have been selected for an exclusive internship. Pay $500 processing fee immediately. Wire transfer only. Guaranteed placement at Fortune 500 company. Act now before spots fill up!",
        "Data entry internship - earn $3000/month from home! No interview required. Just send your bank details and SSN for direct deposit setup. Start immediately!",
        "FREE internship certificate! Just pay $150 for materials. Work from anywhere. No skills required. Guaranteed job placement after completion. WhatsApp us at +91-XXXXXXXXXX.",
        "Earn while you learn! Secret shopper internship. We'll send you a check for $2500, keep $500 as payment and wire the rest back. Easy money!",
        "International internship program. $100 application fee required. No interview needed. Work from home. Earn $4000/month. Send money via Western Union to confirm your spot.",
        "HIRING NOW! Virtual assistant intern needed ASAP. $50/hour guaranteed. No experience. Send personal ID and bank info to get started today. Don't miss this opportunity!",
        "Exclusive unpaid internship at unnamed top company. Must purchase $300 training kit. Certificate guaranteed. No interviews. Apply by sending payment to personal PayPal account.",
        "Congratulations you won internship lottery! Premium internship position available. Just verify identity by sending SSN, DOB, and credit card number. 100% legitimate opportunity.",
        "Work from home social media internship. Make $2000 weekly posting ads. Investment of $250 required for starter package. Money-back guarantee! Contact via personal Gmail only.",
        "AMAZING OPPORTUNITY! Become a brand ambassador intern. Buy our products worth $400 first to understand the brand. Commission-based earnings. Unlimited income potential!",
        "Government internship program - guaranteed placement. Pay $350 registration fee. No eligibility criteria. Contact us on WhatsApp only. Certificate from Ministry of Education included.",
        "Crypto trading internship! Learn to trade and earn $10,000/month. Initial investment of $500 required. 100% returns guaranteed. No prior experience needed. Act fast!",
        "Remote internship at top MNC. Salary: $5000/month. Requirements: None. Just fill form with your Aadhaar number, PAN card, and bank details. Immediate joining.",
        "PAID internship! But first pay $200 for background check and onboarding kit. The company name is confidential until you pay. Trust us - this is a Fortune 100 company!",
        "Freelance writing internship - earn $100 per article. Must first complete unpaid trial of 20 articles to prove your skills. Payment starts after trial period. No contract provided.",
        "IT internship at home! We provide all equipment - just pay $450 shipping and handling fee. Earn $3500/month. No technical skills needed. Diploma holders welcome.",
        "Exclusive research internship. Pay $175 for access to our proprietary research tools. Publication guaranteed in 2 weeks. No mentor supervision needed. Certificate included.",
        "Marketing internship with immediate offer letter! No interview. Pay $100 processing fee. We'll email your offer letter within 24 hours. Share with 10 friends for referral bonus.",
        "Stock market internship. Invest $1000 and we double it in a week while teaching you trading. Guaranteed returns. No risk involved. WhatsApp for more details.",
        "Travel and tourism internship. Pay $600 for all-inclusive training in Dubai. Guaranteed hotel placement. No qualifications needed. Visa and accommodation included (additional charges apply).",
        "App development internship. No coding knowledge needed. Pay $250 for our AI-powered course. Build 10 apps in 5 days. Guaranteed internship certificate. Limited spots!",
        "HR internship at unnamed company. Work 1 hour daily, earn $2000 weekly. Send resume to personal Gmail. Pay $75 for company email access. Start tomorrow!",
        "Paid clinical research internship. No medical background needed. Deposit $300 for lab access card. Earn $4000/month testing new supplements. No risks involved.",
        "Digital marketing internship with Google certification! Pay $180 for study material. Exam answers provided. 100% pass rate. Certificate shipped within 3 days.",
        "Photography internship. Buy our camera package for $800 and join our team. Earn $500 per photoshoot. No experience needed. Equipment is yours to keep!",
        "URGENT HIRING: Customer support intern. $40/hour. Pay $150 for training software license. Work whenever you want. No schedule required. PayPal payment weekly.",
        "Blockchain development internship. Invest 0.5 BTC to access our exclusive training. Return of 2 BTC guaranteed within 30 days plus internship certificate.",
        "NGO internship abroad. Pay $500 for placement and accommodation. No interviews. Tax-deductible. Certificate of volunteering included. Apply through WhatsApp group link.",
        "AI/ML internship. No prerequisites. Pay $200 for GPU cloud access. Build your portfolio in 3 days. Guaranteed referral to Amazon/Google. Money back if not satisfied.",
    ]

    data = []
    for text in genuine_examples:
        data.append({"text": text, "label": 0})
    for text in fraudulent_examples:
        data.append({"text": text, "label": 1})

    df = pd.DataFrame(data)
    # Save locally for future use
    os.makedirs(os.path.join(os.path.dirname(__file__), "data"), exist_ok=True)
    save_path = os.path.join(os.path.dirname(__file__), "data", "dataset.csv")
    df.to_csv(save_path, index=False)
    print(f"Saved synthetic dataset with {len(df)} records to {save_path}")
    return df


def train_and_save_model():
    """Train ML models, evaluate them, and save the best one."""
    print("=" * 60)
    print("Internship Scam Detection - Model Training")
    print("=" * 60)

    # Step 1: Load dataset
    df = load_dataset()
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Identify text and label columns
    text_col = None
    label_col = None

    # Try to find text column
    for col in ["text", "description", "job_description", "content", "posting"]:
        if col in df.columns:
            text_col = col
            break

    # Try to find label column
    for col in ["label", "fraudulent", "is_fake", "is_fraudulent", "fake", "target"]:
        if col in df.columns:
            label_col = col
            break

    # If columns not found, try to combine text columns
    if text_col is None:
        text_columns = df.select_dtypes(include=["object"]).columns.tolist()
        if label_col and label_col in text_columns:
            text_columns.remove(label_col)
        if text_columns:
            df["combined_text"] = df[text_columns].fillna("").agg(" ".join, axis=1)
            text_col = "combined_text"

    if label_col is None:
        # Try numeric columns as potential label
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        for col in numeric_cols:
            if df[col].nunique() == 2:
                label_col = col
                break

    if text_col is None or label_col is None:
        print(f"ERROR: Could not identify text column ({text_col}) or label column ({label_col})")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    print(f"\nUsing text column: '{text_col}'")
    print(f"Using label column: '{label_col}'")
    print(f"\nLabel distribution:\n{df[label_col].value_counts()}")

    # Step 2: Preprocess text
    print("\nPreprocessing text data...")
    df["processed_text"] = df[text_col].fillna("").apply(preprocess_text)

    # Remove empty rows after preprocessing
    df = df[df["processed_text"].str.strip().astype(bool)].reset_index(drop=True)
    print(f"Records after preprocessing: {len(df)}")

    # Step 3: Split data
    X = df["processed_text"]
    y = df[label_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")

    # Step 4: TF-IDF Feature Extraction
    print("\nExtracting TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    print(f"TF-IDF feature matrix: {X_train_tfidf.shape}")

    # Step 5: Train multiple classifiers
    classifiers = {
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "SVM (Linear)": LinearSVC(max_iter=2000, C=1.0, random_state=42),
    }

    best_model = None
    best_name = ""
    best_accuracy = 0

    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train_tfidf, y_train)
        y_pred = clf.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Genuine", "Fraudulent"]))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = clf
            best_name = name

    print(f"\nBest Model: {best_name} (Accuracy: {best_accuracy:.4f})")

    # Step 6: Save model and vectorizer
    save_dir = os.path.join(os.path.dirname(__file__), "saved_model")
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "classifier.joblib")
    vectorizer_path = os.path.join(save_dir, "tfidf_vectorizer.joblib")
    metadata_path = os.path.join(save_dir, "model_metadata.joblib")

    joblib.dump(best_model, model_path)
    joblib.dump(tfidf, vectorizer_path)
    joblib.dump(
        {
            "model_name": best_name,
            "accuracy": best_accuracy,
            "feature_count": X_train_tfidf.shape[1],
            "training_samples": len(X_train),
            "labels": {0: "Genuine", 1: "Fraudulent"},
        },
        metadata_path,
    )

    print(f"\nModel saved to: {model_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")
    print(f"Metadata saved to: {metadata_path}")
    print("\nTraining complete!")


if __name__ == "__main__":
    train_and_save_model()
