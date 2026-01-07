"""
Model evaluation utilities for property address classification.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
import joblib


def evaluate_model(model, X, y_true, label_encoder, print_report=True):
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model
        X: Feature matrix
        y_true: True labels
        label_encoder: Fitted label encoder
        print_report: Whether to print classification report
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    
    # Get class names
    class_names = label_encoder.classes_
    
    # Print classification report
    if print_report:
        print("=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(classification_report(
            y_true, y_pred,
            target_names=class_names,
            digits=4
        ))
        
        print("\n" + "=" * 60)
        print("OVERALL METRICS")
        print("=" * 60)
        print(f"Accuracy:          {accuracy:.4f}")
        print(f"Macro F1 Score:    {macro_f1:.4f}")
        print(f"Macro Precision:   {macro_precision:.4f}")
        print(f"Macro Recall:      {macro_recall:.4f}")
        print("=" * 60)
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'y_pred': y_pred,
        'y_true': y_true
    }


def plot_confusion_matrix(y_true, y_pred, label_encoder, save_path='best_model/confusion_matrix.png'):
    """
    Plot and optionally save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_encoder: Fitted label encoder
        save_path: Path to save the plot (default: best_model/confusion_matrix.png)
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    class_names = label_encoder.classes_
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save or show
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    return cm


def analyze_misclassifications(df, y_true, y_pred, label_encoder, top_n=10):
    """
    Analyze most common misclassifications.
    
    Args:
        df: Original dataframe with addresses
        y_true: True labels
        y_pred: Predicted labels
        label_encoder: Fitted label encoder
        top_n: Number of examples to show per confusion pair
        
    Returns:
        pd.DataFrame: Dataframe with misclassification analysis
    """
    # Get class names
    class_names = label_encoder.classes_
    
    # Create dataframe with predictions
    results_df = df.copy()
    results_df['true_label'] = label_encoder.inverse_transform(y_true)
    results_df['pred_label'] = label_encoder.inverse_transform(y_pred)
    results_df['correct'] = y_true == y_pred
    
    # Filter misclassifications
    misclassified = results_df[~results_df['correct']]
    
    if len(misclassified) == 0:
        print("No misclassifications found!")
        return None
    
    print("=" * 80)
    print("MISCLASSIFICATION ANALYSIS")
    print("=" * 80)
    print(f"\nTotal misclassified: {len(misclassified)} / {len(results_df)}")
    print(f"Misclassification rate: {len(misclassified)/len(results_df)*100:.2f}%\n")
    
    # Group by true/predicted pairs
    confusion_pairs = misclassified.groupby(['true_label', 'pred_label']).size()
    confusion_pairs = confusion_pairs.sort_values(ascending=False)
    
    print("Most common confusion pairs:")
    print("-" * 80)
    for (true, pred), count in confusion_pairs.head(10).items():
        print(f"{true:20s} → {pred:20s} : {count:4d} cases")
    
    return misclassified


def compare_models(models_dict, X, y_true, label_encoder):
    """
    Compare multiple models on the same dataset.
    
    Args:
        models_dict: Dictionary of {model_name: model}
        X: Feature matrix
        y_true: True labels
        label_encoder: Fitted label encoder
        
    Returns:
        pd.DataFrame: Comparison table
    """
    results = []
    
    for model_name, model in models_dict.items():
        y_pred = model.predict(X)
        
        results.append({
            'Model': model_name,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Macro F1': f1_score(y_true, y_pred, average='macro'),
            'Macro Precision': precision_score(y_true, y_pred, average='macro'),
            'Macro Recall': recall_score(y_true, y_pred, average='macro')
        })
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('Macro F1', ascending=False)
    
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80)
    
    return comparison_df


def plot_training_history(history, save_path=None):
    """
    Plot training history for neural network models.
    
    Args:
        history: Training history object (e.g., from Keras)
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontweight='bold')
    axes[0].set_title('Model Accuracy', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel('Loss', fontweight='bold')
    axes[1].set_title('Model Loss', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    else:
        plt.show()


def save_predictions(df, y_pred, label_encoder, save_path):
    """
    Save predictions to CSV file.
    
    Args:
        df: Original dataframe
        y_pred: Predicted labels
        label_encoder: Fitted label encoder
        save_path: Path to save CSV
    """
    results_df = df.copy()
    results_df['predicted_category'] = label_encoder.inverse_transform(y_pred)
    results_df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")


if __name__ == "__main__":
    print("This module provides evaluation utilities.")