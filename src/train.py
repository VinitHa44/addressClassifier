"""
Model training script for property address classification.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import joblib
import argparse
from preprocessing import (
    preprocess_dataframe,
    create_tfidf_features,
    encode_labels,
    save_preprocessors
)


def train_logistic_regression(X_train, y_train, class_weight='balanced'):
    """
    Train Logistic Regression classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        class_weight: Strategy for handling class imbalance
        
    Returns:
        Trained model
    """
    model = LogisticRegression(
        max_iter=1000,
        class_weight=class_weight,
        random_state=42,
        solver='lbfgs'
    )
    
    model.fit(X_train, y_train)
    return model


def train_linear_svm(X_train, y_train, class_weight='balanced'):
    """
    Train Linear SVM classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        class_weight: Strategy for handling class imbalance
        
    Returns:
        Trained model
    """
    model = LinearSVC(
        max_iter=1000,
        class_weight=class_weight,
        random_state=42,
        dual=False
    )
    
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, class_weight='balanced'):
    """
    Train Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        class_weight: Strategy for handling class imbalance
        
    Returns:
        Trained model
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model


def save_model(model, model_name, save_dir='best_model'):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        model_name: Name for the saved model file
        save_dir: Directory to save the model
    """
    filepath = f'{save_dir}/{model_name}.pkl'
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(model_name, save_dir='best_model'):
    """
    Load a saved model from disk.
    
    Args:
        model_name: Name of the saved model file
        save_dir: Directory containing the model
        
    Returns:
        Loaded model
    """
    filepath = f'{save_dir}/{model_name}.pkl'
    model = joblib.load(filepath)
    return model


def main(train_path, val_path, model_type='logistic', save_dir='best_model'):
    """
    Main training pipeline.
    
    Args:
        train_path: Path to training CSV file
        val_path: Path to validation CSV file
        model_type: Type of model to train ('logistic', 'svm', 'rf')
        save_dir: Directory to save models and preprocessors
    """
    print("=" * 50)
    print("Property Address Classification - Training")
    print("=" * 50)
    
    # Load data
    print("\n1. Loading data...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Preprocess text
    print("\n2. Preprocessing text...")
    train_df = preprocess_dataframe(train_df)
    val_df = preprocess_dataframe(val_df)
    
    # Create TF-IDF features
    print("\n3. Creating TF-IDF features...")
    X_train, X_val, _, vectorizer = create_tfidf_features(
        train_df['clean_address'],
        val_df['clean_address']
    )
    print(f"Feature dimensions: {X_train.shape[1]}")
    
    # Encode labels
    print("\n4. Encoding labels...")
    y_train, y_val, _, label_encoder = encode_labels(
        train_df['categories'],
        val_df['categories']
    )
    print(f"Classes: {label_encoder.classes_}")
    
    # Save preprocessors
    print("\n5. Saving preprocessors...")
    save_preprocessors(vectorizer, label_encoder, save_dir)
    
    # Train model
    print(f"\n6. Training {model_type} model...")
    if model_type == 'logistic':
        model = train_logistic_regression(X_train, y_train)
    elif model_type == 'svm':
        model = train_linear_svm(X_train, y_train)
    elif model_type == 'rf':
        model = train_random_forest(X_train, y_train)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Calculate training accuracy
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    
    # Save model
    print("\n7. Saving model...")
    save_model(model, f'{model_type}_model', save_dir)
    
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train address classification model')
    parser.add_argument('--train', type=str, default='data/train.csv',
                        help='Path to training CSV file')
    parser.add_argument('--val', type=str, default='data/val.csv',
                        help='Path to validation CSV file')
    parser.add_argument('--model', type=str, default='logistic',
                        choices=['logistic', 'svm', 'rf'],
                        help='Type of model to train')
    parser.add_argument('--save_dir', type=str, default='best_model',
                        help='Directory to save models')
    
    args = parser.parse_args()
    main(args.train, args.val, args.model, args.save_dir)
