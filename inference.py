"""
Example inference script for making predictions on new addresses.

This script demonstrates how to load saved models and make predictions.
"""

import joblib
import pandas as pd
from src.preprocessing import clean_text


def load_models(model_dir='best_model'):
    """
    Load saved model and preprocessors.
    
    Args:
        model_dir: Directory containing saved models (default: best_model)
        
    Returns:
        tuple: (model, vectorizer, label_encoder)
    """
    model = joblib.load(f'{model_dir}/best_model.pkl')
    vectorizer = joblib.load(f'{model_dir}/tfidf_vectorizer.pkl')
    label_encoder = joblib.load(f'{model_dir}/label_encoder.pkl')
    
    return model, vectorizer, label_encoder


def predict_single_address(address, model, vectorizer, label_encoder):
    """
    Predict category for a single address.
    
    Args:
        address: Raw address string
        model: Trained classifier
        vectorizer: Fitted TF-IDF vectorizer
        label_encoder: Fitted label encoder
        
    Returns:
        str: Predicted category
    """
    # Preprocess
    cleaned = clean_text(address)
    
    # Transform to features
    features = vectorizer.transform([cleaned])
    
    # Predict
    prediction = model.predict(features)[0]
    
    # Decode to category name
    category = label_encoder.inverse_transform([prediction])[0]
    
    return category


# def predict_batch_addresses(addresses, model, vectorizer, label_encoder):
#     """
#     Predict categories for multiple addresses.
    
#     Args:
#         addresses: List of raw address strings
#         model: Trained classifier
#         vectorizer: Fitted TF-IDF vectorizer
#         label_encoder: Fitted label encoder
        
#     Returns:
#         list: List of predicted categories
#     """
#     # Preprocess all addresses
#     cleaned_addresses = [clean_text(addr) for addr in addresses]
    
#     # Transform to features
#     features = vectorizer.transform(cleaned_addresses)
    
#     # Predict
#     predictions = model.predict(features)
    
#     # Decode to category names
#     categories = label_encoder.inverse_transform(predictions)
    
#     return categories.tolist()


# def predict_from_csv(input_csv, output_csv, model, vectorizer, label_encoder,
#                      address_column='property_address'):
#     """
#     Make predictions for addresses in a CSV file.
    
#     Args:
#         input_csv: Path to input CSV file
#         output_csv: Path to save output CSV file
#         model: Trained classifier
#         vectorizer: Fitted TF-IDF vectorizer
#         label_encoder: Fitted label encoder
#         address_column: Name of column containing addresses
#     """
#     # Load data
#     df = pd.read_csv(input_csv)
    
#     # Make predictions
#     predictions = predict_batch_addresses(
#         df[address_column].tolist(),
#         model,
#         vectorizer,
#         label_encoder
#     )
    
#     # Add predictions to dataframe
#     df['predicted_category'] = predictions
    
#     # Save results
#     df.to_csv(output_csv, index=False)
#     print(f"Predictions saved to {output_csv}")
    
#     return df


def main():
    print("Property Address Classification")
    print("="*80)
    
    # Load models
    print("\nLoading models...")
    try:
        model, vectorizer, label_encoder = load_models()
        print("✓ Models loaded successfully")
    except FileNotFoundError:
        print("❌ Error: Model files not found in 'best_model/' folder.")
        print("Please train the model first by running notebooks/02_modeling.ipynb")
        return
    
    print("\nAvailable categories:")
    for i, cat in enumerate(label_encoder.classes_, 1):
        print(f"  {i}. {cat}")
    
    print("\n" + "-" * 80)
    print("Enter property addresses to classify (type 'quit' or 'exit' to stop)")
    print("-" * 80)
    
    while True:
        # Get user input
        address = input("\nEnter address: ").strip()
        
        # Check for exit commands
        if address.lower() in ['quit', 'exit', 'q']:
            print("\n" + "="*80)
            print("Thank you for using Property Address Classifier!")
            print("="*80)
            break
        
        # Skip empty inputs
        if not address:
            print("⚠️  Please enter a valid address")
            continue
        
        # Make prediction
        try:
            category = predict_single_address(address, model, vectorizer, label_encoder)
            print(f"✓ Predicted Category: {category}")
        except Exception as e:
            print(f"❌ Error making prediction: {str(e)}")


if __name__ == "__main__":
    main()
