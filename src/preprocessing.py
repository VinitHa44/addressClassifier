"""
Text preprocessing and feature extraction utilities for address classification.
"""

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib


def clean_text(text):
    """
    Clean and normalize address text.
    
    Args:
        text (str): Raw address text
        
    Returns:
        str: Cleaned address text
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters but keep numbers and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def preprocess_dataframe(df, text_column='property_address'):
    """
    Apply text cleaning to entire dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Name of column containing addresses
        
    Returns:
        pd.DataFrame: Dataframe with cleaned text column
    """
    df = df.copy()
    df['clean_address'] = df[text_column].apply(clean_text)
    return df


def create_tfidf_features(train_texts, val_texts=None, test_texts=None,
                          ngram_range=(1, 2), min_df=2, max_features=5000):
    """
    Create TF-IDF features from text data.
    
    Args:
        train_texts: Training text data
        val_texts: Validation text data (optional)
        test_texts: Test text data (optional)
        ngram_range: Range of n-grams (default: unigrams and bigrams)
        min_df: Minimum document frequency
        max_features: Maximum number of features
        
    Returns:
        tuple: (X_train, X_val, X_test, vectorizer)
    """
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        strip_accents='unicode',
        lowercase=True
    )
    
    # Fit on training data
    X_train = vectorizer.fit_transform(train_texts)
    
    # Transform validation and test data if provided
    X_val = vectorizer.transform(val_texts) if val_texts is not None else None
    X_test = vectorizer.transform(test_texts) if test_texts is not None else None
    
    return X_train, X_val, X_test, vectorizer


def encode_labels(train_labels, val_labels=None, test_labels=None):
    """
    Encode categorical labels to numeric format.
    
    Args:
        train_labels: Training labels
        val_labels: Validation labels (optional)
        test_labels: Test labels (optional)
        
    Returns:
        tuple: (y_train, y_val, y_test, label_encoder)
    """
    label_encoder = LabelEncoder()
    
    # Fit on training data
    y_train = label_encoder.fit_transform(train_labels)
    
    # Transform validation and test data if provided
    y_val = label_encoder.transform(val_labels) if val_labels is not None else None
    y_test = label_encoder.transform(test_labels) if test_labels is not None else None
    
    return y_train, y_val, y_test, label_encoder


def save_preprocessors(vectorizer, label_encoder, save_dir='best_model'):
    """
    Save vectorizer and label encoder for reproducibility.
    
    Args:
        vectorizer: Fitted TF-IDF vectorizer
        label_encoder: Fitted label encoder
        save_dir: Directory to save the preprocessors
    """
    joblib.dump(vectorizer, f'{save_dir}/tfidf_vectorizer.pkl')
    joblib.dump(label_encoder, f'{save_dir}/label_encoder.pkl')
    print(f"Preprocessors saved to {save_dir}/")


def load_preprocessors(save_dir='best_model'):
    """
    Load saved preprocessors.
    
    Args:
        save_dir: Directory containing saved preprocessors
        
    Returns:
        tuple: (vectorizer, label_encoder)
    """
    vectorizer = joblib.load(f'{save_dir}/tfidf_vectorizer.pkl')
    label_encoder = joblib.load(f'{save_dir}/label_encoder.pkl')
    return vectorizer, label_encoder
