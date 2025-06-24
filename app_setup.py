# app_setup.py (Corrected Version)

"""
One-time setup script to train an OPTIMIZED model and prepare artifacts.
This version uses GridSearchCV to find the best model settings.
"""
import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV

# Import the cleaning function
from ticket_classifier import advanced_clean_text

# --- Configuration ---
# <<< CHANGE THIS LINE
DATA_FILE_PATH = 'data/master_training_data.csv' # Use the new combined dataset
MODEL_DIR = 'models'
DOCUMENT_COL = 'Document'
TOPIC_COL = 'Topic_group' # This is the correct variable name we should use
MISC_LABEL = 'Miscellaneous'
NUM_CLUSTERS = 8

def setup():
    """Trains and saves all necessary models and artifacts."""
    print("--- Starting Application Setup (with Model Optimization) ---")
    
    os.makedirs(MODEL_DIR, exist_ok=True)

    try:
        df = pd.read_csv(DATA_FILE_PATH)
        print(f"Loaded {len(df)} records from '{DATA_FILE_PATH}'.")
    except FileNotFoundError:
        print(f"FATAL: Data file not found at '{DATA_FILE_PATH}'.")
        print("Please make sure you have run 'python prepare_data.py' first.")
        return

    print("Cleaning text data...")
    df['clean'] = df[DOCUMENT_COL].apply(advanced_clean_text)

    # Separate data for training
    core_df = df[df[TOPIC_COL] != MISC_LABEL].copy()
    
    # --- MODEL IMPROVEMENT: Define a pipeline and a grid of parameters to search ---
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', SGDClassifier(random_state=42)) 
    ])

    parameters = [
        # Grid for SGDClassifier
        {
            'tfidf__max_df': (0.75, 1.0),
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'clf': [SGDClassifier(loss='hinge', penalty='l2', random_state=42)],
            'clf__alpha': (1e-4, 1e-5),
        },
        # Grid for LogisticRegression
        {
            'tfidf__max_df': (0.75, 1.0),
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'clf': [LogisticRegression(solver='liblinear', random_state=42)],
            'clf__C': (0.1, 1.0, 10.0), # 'C' is the regularization parameter for LogisticRegression
        },
    ]

    print("\nStarting hyperparameter tuning with GridSearchCV...")
    print("This may take a few minutes depending on your computer...")
    
    grid_search = GridSearchCV(pipeline, parameters, cv=3, n_jobs=-1, verbose=2)
    
    grid_search.fit(core_df["clean"], core_df[TOPIC_COL])

    print("\nBest parameters found:")
    print(grid_search.best_params_)
    
    best_pipeline = grid_search.best_estimator_

    pipeline_path = os.path.join(MODEL_DIR, 'ticket_classifier.joblib')
    joblib.dump(best_pipeline, pipeline_path)
    print(f"\n✅ Optimized classifier saved to '{pipeline_path}'")

    # The rest of the script remains the same
    print(f"\nIsolating and clustering '{MISC_LABEL}' tickets...")
    misc_df = df[df[TOPIC_COL] == MISC_LABEL].copy()

    if len(misc_df) >= NUM_CLUSTERS:
        tfidf_misc = best_pipeline.named_steps["tfidf"].transform(misc_df['clean'])
        kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
        kmeans.fit(tfidf_misc)
        
        kmeans_path = os.path.join(MODEL_DIR, 'kmeans_model.joblib')
        joblib.dump(kmeans, kmeans_path)
        print(f"✅ KMeans model saved to '{kmeans_path}'")

        terms = best_pipeline.named_steps["tfidf"].get_feature_names_out()
        terms_path = os.path.join(MODEL_DIR, 'tfidf_terms.joblib')
        joblib.dump(terms, terms_path)
        print(f"✅ TF-IDF terms saved to '{terms_path}'")
    else:
        print("Warning: Not enough miscellaneous documents to create clusters. Skipping.")
        
    print("\n--- Setup Complete ---")

if __name__ == "__main__":
    setup()