# train_and_evaluate.py (Final Version)

import os
import re
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# --- Configuration ---
# --- CHANGE: Point to the new final training file ---
DATA_FILE_PATH = 'data/IT_tickets_final_training.csv'
MODEL_DIR = 'models'
DOCUMENT_COL = 'Document'
TOPIC_COL = 'Topic_group'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# We still use the cleaning function, as it helps focus the model
def clean_ticket_body(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'from:.*|sent:.*|to:.*|subject:.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'hi,?|hello,?|dear,?|regards,?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'thanks?|best regards|kind regards', '', text, flags=re.IGNORECASE)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def train_and_evaluate_deep_model():
    """Main function to train the final classifier."""
    
    # 1. Load Data
    try:
        df = pd.read_csv(DATA_FILE_PATH)
        print(f"Loaded {len(df)} records from '{DATA_FILE_PATH}'.")
    except FileNotFoundError:
        print(f"FATAL: Master data file not found at '{DATA_FILE_PATH}'.")
        print("Please run 'python prepare_final_data.py' first.")
        return

    # 2. Prepare and Clean Data
    print("\n--- Step 1: Preparing and Cleaning Data ---")
    df.dropna(subset=[DOCUMENT_COL, TOPIC_COL], inplace=True)
    print("Applying text cleaning...")
    df['clean_doc'] = df[DOCUMENT_COL].apply(clean_ticket_body)
    
    # --- CHANGE: We no longer separate Miscellaneous. We train on all data. ---
    core_df = df.copy() 
    
    print(f"Training on {len(core_df)} tickets with the following category distribution:")
    print(core_df[TOPIC_COL].value_counts())

    # 3. Generate Embeddings
    print(f"\n--- Step 2: Loading SentenceTransformer model '{EMBEDDING_MODEL_NAME}' ---")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("\n--- Step 3: Generating Embeddings for Training Data (this may take a while) ---")
    X = embedding_model.encode(core_df['clean_doc'].tolist(), show_progress_bar=True)
    y = core_df[TOPIC_COL]
    
    # 4. Train Classifier
    print("\n--- Step 4: Training Classifier on Embeddings ---")
    param_grid = {'C': [0.1, 1, 10, 100]}
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

    # 6. Save Models
    print("\n--- Step 6: Saving Final Classifier Model ---")
    os.makedirs(MODEL_DIR, exist_ok=True)
    classifier_path = os.path.join(MODEL_DIR, 'ticket_classifier_deep.joblib')
    joblib.dump(best_classifier, classifier_path)
    print(f"✅ Trained classifier saved to '{classifier_path}'")
    print("\n--- Full Deep Learning Training and Evaluation Complete ---")

if __name__ == "__main__":
    train_and_evaluate_deep_model()