"""
Quick script to display actual performance metrics of the best model.
"""

import joblib
import pandas as pd
from src.preprocessing import preprocess_dataframe, create_tfidf_features, encode_labels
from src.evaluate import evaluate_model, plot_confusion_matrix

# Load validation data
print("Loading validation data...")
val_df = pd.read_csv('data/val.csv')

# Preprocess
print("Preprocessing...")
val_df = preprocess_dataframe(val_df)

# Load saved models and preprocessors
print("Loading trained models...")
model = joblib.load('best_model/best_model.pkl')
vectorizer = joblib.load('best_model/tfidf_vectorizer.pkl')
label_encoder = joblib.load('best_model/label_encoder.pkl')

# Transform features
print("Transforming features...")
X_val = vectorizer.transform(val_df['clean_address'])
y_val = label_encoder.transform(val_df['categories'])

# Evaluate
print("\n" + "="*70)
print("ACTUAL PERFORMANCE METRICS - BEST MODEL")
print("="*70 + "\n")

results = evaluate_model(model, X_val, y_val, label_encoder, print_report=True)

# Plot confusion matrix
print("\nGenerating confusion matrix...")
plot_confusion_matrix(
    results['y_true'], 
    results['y_pred'], 
    label_encoder,
    save_path='best_model/confusion_matrix_validation.png'
)

print("\n✅ Confusion matrix saved to: best_model/confusion_matrix_validation.png")
print("\n" + "="*70)
print(f"SUMMARY - Macro F1 Score: {results['macro_f1']:.4f}")
print("="*70)
