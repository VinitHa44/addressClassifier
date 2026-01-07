# 🏠 Property Address Classification Project

A machine learning project for classifying property addresses into predefined categories using traditional NLP and ML techniques.

## 📋 Table of Contents
- [Problem Statement](#problem-statement)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Approach](#approach)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Reproducibility](#reproducibility)

---

## 🎯 Problem Statement

**Goal**: Build a classifier that accurately classifies property addresses into predefined categories.

**Predefined Categories**:
- `flat`
- `houseorplot`
- `landparcel`
- `commercial unit`
- `others`

**Input**: Raw property address text (string)  
**Output**: Category label (string)

**Constraints**:
- Using only prompt engineering for classification is discouraged
- Clean reasoning, clear methodology, and reproducibility are important
- Focus on achieving reasonable performance with sound approach

---

## 📁 Project Structure

```
property_classifier/
│
├── data/                      # Dataset files
│   ├── train.csv             # Training dataset
│   └── val.csv               # Validation dataset
│
├── notebooks/                 # Jupyter notebooks
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   └── 02_modeling.ipynb     # Model training and evaluation
│
├── src/                       # Source code
│   ├── preprocessing.py      # Text preprocessing utilities
│   ├── train.py              # Model training script
│   └── evaluate.py           # Evaluation utilities
│
├── best_model/                # Best model artifacts (REQUIRED)
│   ├── best_model.pkl        # Best trained model
│   ├── tfidf_vectorizer.pkl  # TF-IDF vectorizer
│   ├── label_encoder.pkl     # Label encoder
│   └── confusion_matrix.png  # Confusion matrix visualization
│
├── models/                    # Additional models directory
│
├── approach.txt               # Detailed approach documentation (REQUIRED)
├── requirements.txt           # Python dependencies
├── inference.py               # Prediction script
└── README.md                  # This file
```

---

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- VS Code (recommended for running notebooks)

### Quick Setup

**For complete step-by-step instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md)**

### Basic Setup

1. **Open project in VS Code**:
   ```bash
   cd addressClassifier
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1  # Windows
   # source venv/bin/activate    # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python -c "import sklearn, pandas, numpy; print('All packages installed successfully!')"
   ```

---

## 📊 Dataset

### Dataset Format

Both `train.csv` and `val.csv` should contain:
- **property_address** (str): Raw address text
- **categories** (str): Label from predefined categories

### Example:
```csv
property_address,categories
"Flat No 302, Sai Residency, Near Axis Bank",flat
"Plot No 45, Survey No 123, Open Land",landparcel
"Shop No 12, Commercial Complex, MG Road",commercial unit
```

### Place Dataset Files
1. Download your training and validation datasets
2. Place them in the `data/` directory:
   - `data/train.csv`
   - `data/val.csv`

---

## 🧠 Approach

### Why Traditional ML + NLP?

This project uses **Traditional Machine Learning with NLP** techniques because:
- ✅ **Interpretable**: Easy to understand model decisions
- ✅ **Reproducible**: Consistent results with saved parameters
- ✅ **Strong Baseline**: Excellent performance for text classification
- ✅ **Fits Requirements**: Aligns with assignment expectations

### Pipeline Overview

```
Raw Address Text
      ↓
Text Preprocessing (lowercase, remove special chars, keep numbers)
      ↓
TF-IDF Feature Extraction (unigrams + bigrams)
      ↓
Model Training (Logistic Regression / SVM / Random Forest)
      ↓
Evaluation (Classification Report, Confusion Matrix)
```

### Text Preprocessing
1. Convert to lowercase
2. Remove special characters (keep numbers - important for addresses!)
3. Remove extra whitespace
4. Preserve meaningful tokens (plot numbers, flat numbers, etc.)

### Feature Extraction
- **Method**: TF-IDF (Term Frequency - Inverse Document Frequency)
- **N-grams**: Unigrams and bigrams (1,2)
- **Max Features**: 5000
- **Min Document Frequency**: 2

### Models Trained
1. **Logistic Regression** (baseline) - with balanced class weights
2. **Linear SVM** - for linear decision boundaries
3. **Random Forest** (optional) - for non-linear patterns

---

## 🚀 Usage

**📘 For complete setup and running instructions for reviewers/evaluators, see [SETUP_GUIDE.md](SETUP_GUIDE.md)**

### Option 1: Using Jupyter Notebooks in VS Code (Recommended)

#### Step 1: Install Jupyter Extension
1. Open VS Code Extensions (Ctrl+Shift+X)
2. Search "Jupyter" and install the extension by Microsoft

#### Step 2: Run Notebooks Interactively
```bash
jupyter notebook notebooks/01_eda.ipynb
```

**What it does**:
- Loads and inspects datasets
- Analyzes class distribution
- Examines text characteristics
- Identifies important keywords per category
- Provides insights for modeling

#### Step 2: Model Training & Evaluation
```bash
jupyter notebook notebooks/02_modeling.ipynb
```

**What it does**:
- Preprocesses text data
- Extracts TF-IDF features
- Trains multiple models
- Compares model performance
- Generates evaluation metrics
- Saves best model

### Option 2: Using Python Scripts

#### Train a Model
```bash
python src/train.py --train data/train.csv --val data/val.csv --model logistic
```

**Arguments**:
- `--train`: Path to training CSV
- `--val`: Path to validation CSV
- `--model`: Model type (`logistic`, `svm`, `rf`)
- `--save_dir`: Directory to save models (default: `models`)

#### Example: Train All Models
```bash
# Train Logistic Regression
python src/train.py --model logistic

# Train Linear SVM
python src/train.py --model svm

# Train Random Forest
python src/train.py --model rf
```

### Option 3: Making Predictions (Inference)

After training a model, use the inference script to classify new addresses:

```bash
python inference.py
```

**Interactive Mode**:
```
Enter property addresses to classify (type 'quit' or 'exit' to stop)
--------------------------------------------------------------------------------

Enter address: Flat No 302, Sai Residency, Near Axis Bank
✓ Predicted Category: flat

Enter address: Plot No 45, Survey No 123, Open Land
✓ Predicted Category: landparcel

Enter address: Shop No 12, Commercial Complex, MG Road
✓ Predicted Category: commercial unit
```

---

## 📈 Model Evaluation

### Primary Metrics

The model is evaluated using:

1. **Macro F1 Score** ⭐ (Most Important)
   - Average F1 across all classes
   - Handles class imbalance well
   - Primary metric for model selection

2. **Accuracy**
   - Overall correct predictions

3. **Precision** (Macro Average)
   - How many predicted labels are correct

4. **Recall** (Macro Average)
   - How many actual labels were found

### Evaluation Reports

The evaluation includes:
- ✅ **Classification Report**: Per-class precision, recall, F1
- ✅ **Confusion Matrix**: Visual representation of predictions
- ✅ **Misclassification Analysis**: Common error patterns
- ✅ **Feature Importance**: Top keywords per category (for Logistic Regression)

### Why Macro F1?

When classes are imbalanced, **Macro F1 Score** is crucial because:
- Treats all classes equally
- Doesn't favor majority classes
- Better reflects real-world performance

### Get Actual Model Metrics

To view the actual performance of your trained model on validation data:

```bash
python get_metrics.py
```

**What it does**:
- Loads the best trained model from `models/` or `best_model/`
- Evaluates on validation dataset
- Displays comprehensive classification report
- Shows per-class precision, recall, and F1 scores
- Generates and saves confusion matrix
- Reports final Macro F1 Score

**Example Output**:
```
==================================================================
ACTUAL PERFORMANCE METRICS - BEST MODEL
==================================================================

              precision    recall  f1-score   support

        flat       0.88      0.92      0.90       150
houseorplot       0.85      0.81      0.83       120
  landparcel       0.79      0.83      0.81       100
commercial unit    0.91      0.87      0.89        95
      others       0.82      0.78      0.80        85

    accuracy                           0.85       550
   macro avg       0.85      0.84      0.85       550
weighted avg       0.85      0.85      0.85       550

✅ Confusion matrix saved to: models/confusion_matrix_validation.png

==================================================================
SUMMARY - Macro F1 Score: 0.8463
==================================================================
```

---

## 📊 Results

### Expected Performance

Based on the methodology:

| Metric | Expected Range |
|--------|----------------|
| **Macro F1 Score** | 0.75 - 0.90 |
| **Accuracy** | 0.80 - 0.92 |
| **Macro Precision** | 0.75 - 0.90 |
| **Macro Recall** | 0.75 - 0.90 |

*Actual results depend on dataset quality and size*

### Model Comparison Example

```
Model                  Accuracy  Macro F1  Macro Precision  Macro Recall
Logistic Regression      0.87      0.85         0.86            0.84
Linear SVM               0.86      0.84         0.85            0.83
Random Forest            0.84      0.81         0.82            0.80
```

### Saved Outputs

After training, you'll find:
- `best_model/best_model.pkl` - Best performing model
- `best_model/tfidf_vectorizer.pkl` - Feature extractor
- `best_model/label_encoder.pkl` - Label decoder
- `best_model/confusion_matrix.png` - Confusion matrix visualization
- `approach.txt` - Detailed methodology documentation

---

## 👨‍💻 Author

Property Address Classification Project  
Created for demonstrating text classification best practices

---

## 🤝 Contributing

To improve this project:
1. Add more preprocessing techniques
2. Experiment with different models
3. Enhance visualization
4. Add unit tests
5. Improve documentation

---

**Happy Classifying! 🎉**
