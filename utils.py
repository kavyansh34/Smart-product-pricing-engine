"""
utils.py — Helper Functions for Smart Product Pricing Challenge
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# =========================================================
# 📦 1. Basic Utility Functions
# =========================================================

def clean_text(text):
    """Clean catalog text."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator == 0, 1e-10, denominator)
    return np.mean(np.abs(y_pred - y_true) / denominator) * 100


def clip_outliers(df, col="price", upper_quantile=0.99):
    """Clip extreme outliers in target variable."""
    upper_limit = df[col].quantile(upper_quantile)
    df[col] = df[col].clip(upper=upper_limit)
    print(f"Clipped {col} above {upper_limit:.2f}")
    return df

# =========================================================
# 🧠 2. Text Feature Utilities
# =========================================================

def compute_sentiment(text):
    """Return (polarity, subjectivity) sentiment scores."""
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    except:
        return 0.0, 0.0


def generate_tfidf_features(train_texts, test_texts, max_features=6000):
    """Generate TF-IDF + SVD (LSA) reduced features."""
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), stop_words="english")
    train_tfidf = tfidf.fit_transform(train_texts)
    test_tfidf = tfidf.transform(test_texts)

    print(f"TF-IDF: {train_tfidf.shape}")

    svd = TruncatedSVD(n_components=120, random_state=42)
    train_lsa = svd.fit_transform(train_tfidf)
    test_lsa = svd.transform(test_tfidf)
    print(f"LSA variance explained: {svd.explained_variance_ratio_.sum():.2%}")

    return train_tfidf, test_tfidf, train_lsa, test_lsa

# =========================================================
# 🖼️ 3. Image Feature Extraction (Simple & Cached)
# =========================================================

def extract_image_features(image_paths, cache_file=None):
    """
    Simple handcrafted image feature extraction (color histograms + texture).
    """
    if cache_file and os.path.exists(cache_file):
        print(f"📂 Loading cached features from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    features = []
    for img_path in tqdm(image_paths, desc="Extracting image features"):
        try:
            img = cv2.imread(img_path)
            if img is None:
                features.append(np.zeros(128))
                continue
            img = cv2.resize(img, (128, 128))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [128], [0, 256])
            feats = np.hstack([hist.flatten(), gray.mean(), gray.std()])
            features.append(feats[:128])
        except:
            features.append(np.zeros(128))
    features = np.array(features)

    if cache_file:
        with open(cache_file, "wb") as f:
            pickle.dump(features, f)
    print(f"✅ Extracted {features.shape[1]} features per image.")
    return features

# =========================================================
# 🔧 4. Miscellaneous Helpers
# =========================================================

def save_pickle(obj, filename):
    """Save object to pickle file."""
    with open(filename, "wb") as f:
        pickle.dump(obj, f)
    print(f"💾 Saved to {filename}")

def load_pickle(filename):
    """Load object from pickle file."""
    with open(filename, "rb") as f:
        return pickle.load(f)

def ensure_dir(path):
    """Ensure directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)
