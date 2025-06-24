# train_and_evaluate.py (Deep Learning Version)

"""
Trains a superior classification model using sentence-transformer embeddings.

This version replaces the TF-IDF approach with deep learning embeddings to
capture semantic meaning, leading to significantly better classification.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- NEW: Import SentenceTransformer ---
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# --- Configuration ---
DATA_FILE_PATH = 'data/IT_tickets.csv'
MODEL_DIR = 'models'
DOCUMENT_COL = 'Document'
TOPIC_COL = 'Topic_group'
MISC_LABEL = 'Miscellaneous'
# --- NEW: Define the transformer model we'll use ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# We no longer need the refine_categories or advanced_clean_text functions for this approach,
# as the transformer model handles raw text and its nuances much better. We will train on the original labels.

def train_and_evaluate_deep_model():
    """Main function to train, evaluate, and save the deep learning-based model."""
    
    # 1. Load Data
    try:
        df = pd.read_csv(DATA_FILE_PATH)
        print(f"Loaded {len(df)} records from '{DATA_FILE_PATH}'.")
    except FileNotFoundError:
        print(f"FATAL: Master data file not found at '{DATA_FILE_PATH}'.")
        print("Please run 'python prepare_data.py' first.")
        return

    # 2. Prepare Data
    print("\n--- Step 1: Preparing Data ---")
    # We will use the original text directly. Transformer models are good at this.
    # Let's drop any rows with missing text to be safe.
    df.dropna(subset=[DOCUMENT_COL, TOPIC_COL], inplace=True)
    
    # We still separate the 'Miscellaneous' tickets from the core training data.
    core_df = df[df[TOPIC_COL] != MISC_LABEL].copy()
    
    # Check if there's enough data to train
    if len(core_df) < 100:
        print("FATAL: Not enough labeled data to train a model. Need at least 100 core tickets.")
        return
        
    print(f"Training on {len(core_df)} tickets with the following category distribution:")
    print(core_df[TOPIC_COL].value_counts())

    # 3. Generate Embeddings
    print(f"\n--- Step 2: Loading SentenceTransformer model '{EMBEDDING_MODEL_NAME}' ---")
    # This will download the model from the internet on the first run.
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("\n--- Step 3: Generating Embeddings for Training Data (this may take a while) ---")
    # We encode the 'Document' column directly.
    X = embedding_model.encode(core_df[DOCUMENT_COL].tolist(), show_progress_bar=True)
    y = core_df[TOPIC_COL]
    
    # 4. Train Classifier
    print("\n--- Step 4: Training Classifier on Embeddings ---")
    # We'll use a simple but effective Logistic Regression classifier.
    # GridSearchCV helps find the best hyperparameter (C).
    param_grid = {'C': [0.1, 1, 10, 100]}
    # Using class_weight='balanced' is still very important!
    clf = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X, y)
    
    best_classifier = grid_search.best_estimator_
    print("\nBest classifier parameters found:")
    print(grid_search.best_params_)

    # 5. Evaluate
    print("\n--- Step 5: Evaluating Best Model ---")
    y_pred = best_classifier.predict(X)
    report = classification_report(y, y_pred)
    print("\nClassification Report:")
    print(report)

    # Save Confusion Matrix plot
    fig, ax = plt.subplots(figsize=(15, 15))
    ConfusionMatrixDisplay.from_estimator(best_classifier, X, y, ax=ax, xticks_rotation='vertical')
    plt.title("Confusion Matrix for Best Classifier (on Embeddings)")
    plt.tight_layout()
    # Ensure the models directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix_deep.png'))
    print("\n✅ Confusion matrix plot saved to 'models/confusion_matrix_deep.png'")
    
    # 6. Save the trained classifier
    print("\n--- Step 6: Saving Final Classifier Model ---")
    # IMPORTANT: We only save our trained Logistic Regression classifier.
    # The SentenceTransformer model is loaded from the library directly in the app.
    classifier_path = os.path.join(MODEL_DIR, 'ticket_classifier_deep.joblib')
    joblib.dump(best_classifier, classifier_path)
    print(f"✅ Trained classifier saved to '{classifier_path}'")

    print("\n--- Full Deep Learning Training and Evaluation Complete ---")


if __name__ == "__main__":
    train_and_evaluate_deep_model()