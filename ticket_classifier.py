# ticket_classifier.py

"""
IT Service Ticket Classifier and Clustering Tool

This script provides a complete workflow for training a ticket classification model
and using it to predict categories for new tickets. It includes a mechanism to
identify out-of-distribution ("Unknown") tickets and cluster them to find new patterns.

The script operates in two modes:
1. train: Trains a model on a labeled CSV file and saves the model.
   - It expects a CSV with at least a 'Document' and 'Topic_group' column.
   - The model is trained on all topics EXCEPT for a specified 'miscellaneous' label.
   - The trained pipeline (TF-IDF vectorizer + SGD Classifier) is saved to a file.

2. predict: Loads a pre-trained model and predicts categories for a new (unlabeled) CSV file.
   - It applies the model and a confidence threshold to classify tickets.
   - Tickets below the threshold are marked as 'Unknown'.
   - 'Unknown' tickets are then clustered using KMeans to identify potential new categories.
   - The final output is a new CSV file containing the original data plus prediction and cluster information.

------------------------------------------------------------------------------------------
HOW TO USE:

1. Train a new model:
   python ticket_classifier.py train \
       --data-path "data/all_tickets_processed_improved_v3.csv" \
       --model-path "ticket_classifier.joblib"

2. Predict on a new dataset:
   python ticket_classifier.py predict \
       --data-path "data/new_unlabeled_tickets.csv" \
       --model-path "ticket_classifier.joblib" \
       --output-path "data/predictions_with_clusters.csv" \
       --threshold 0.5
------------------------------------------------------------------------------------------
"""

import argparse
import os
import re
import joblib
import pandas as pd
import numpy as np
import spacy
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import KMeans
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# --- Global Variables & Pre-computation ---
# Download necessary NLTK and spaCy data if not present
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model 'en_core_web_sm' not found. Downloading...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

for resource_id, resource_path in [('stopwords', 'corpora/stopwords'), ('wordnet', 'corpora/wordnet')]:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        print(f"NLTK resource '{resource_id}' not found. Downloading...")
        nltk.download(resource_id, quiet=True)

lemmatizer = WordNetLemmatizer()
base_stop_words = set(stopwords.words('english'))
custom_stop_words = {
    'hi', 'hello', 'thanks', 'thank', 'regards', 'dear', 'pm', 'am', 'sent',
    'please', 'kindly', 'regard', 'best', 'team', 'subject', 're', 'gb', 'mb'
}
STOP_WORDS = base_stop_words.union(custom_stop_words)
MISC_LABEL = "Miscellaneous"


# --- Helper Function for Text Cleaning ---
def advanced_clean_text(text):
    """Applies advanced cleaning: lemmatization and custom stop words."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    tokens = [lemmatizer.lemmatize(word) for word in text.split()]
    tokens = [word for word in tokens if word not in STOP_WORDS and len(word) > 2]
    
    return " ".join(tokens)

# --- Core Logic Functions ---

def train_model(data_path, model_path, doc_col, topic_col):
    """
    Trains the classification model and saves it to a file.
    
    Args:
        data_path (str): Path to the labeled training CSV file.
        model_path (str): Path where the trained model will be saved.
        doc_col (str): Name of the column containing the ticket text.
        topic_col (str): Name of the column containing the topic labels.
    """
    print(f"Loading training data from '{data_path}'...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Training file not found at '{data_path}'. Aborting.")
        return

    print("Cleaning text data...")
    df['clean'] = df[doc_col].apply(advanced_clean_text)

    print(f"Separating core data (excluding '{MISC_LABEL}')...")
    core_df = df[df[topic_col] != MISC_LABEL].copy()
    
    if core_df.empty:
        print(f"Error: No data available for training after excluding '{MISC_LABEL}'. Aborting.")
        return

    print("Training the classification pipeline...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_df=0.8, min_df=5, ngram_range=(1, 2))),
        ("clf", SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, random_state=42, max_iter=1000, tol=1e-3))
    ])
    pipeline.fit(core_df["clean"], core_df[topic_col])

    print(f"Saving the trained model to '{model_path}'...")
    joblib.dump(pipeline, model_path)
    print("\n--- Training complete. Model is saved and ready for prediction. ---")


def predict_and_cluster(data_path, model_path, output_path, doc_col, threshold, num_clusters):
    """
    Loads a model to predict categories and cluster unknown tickets.
    
    Args:
        data_path (str): Path to the new (unlabeled) CSV file.
        model_path (str): Path to the pre-trained model file.
        output_path (str): Path to save the output CSV with predictions.
        doc_col (str): Name of the column containing the ticket text.
        threshold (float): Confidence score threshold for classifying as 'Unknown'.
        num_clusters (int): The number of clusters to create from 'Unknown' tickets.
    """
    print(f"Loading pre-trained model from '{model_path}'...")
    try:
        pipeline = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'. Aborting.")
        return

    print(f"Loading new data from '{data_path}'...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{data_path}'. Aborting.")
        return

    print("Cleaning text data for prediction...")
    df['clean'] = df[doc_col].apply(advanced_clean_text)

    print("Applying model to get predictions and confidence scores...")
    df['predicted_initial'] = pipeline.predict(df['clean'])
    all_scores = pipeline.decision_function(df['clean'])
    df['max_score'] = all_scores.max(axis=1)

    print(f"Applying confidence threshold of {threshold} to identify 'Unknown' tickets...")
    df['predicted_final'] = np.where(
        df['max_score'] >= threshold,
        df['predicted_initial'],
        "Unknown"
    )
    
    unknown_df = df[df['predicted_final'] == 'Unknown'].copy()
    print(f"Found {len(unknown_df)} 'Unknown' tickets to be clustered.")

    df['cluster_id'] = -1  # Default to -1 (no cluster)
    if len(unknown_df) >= num_clusters:
        print(f"Clustering 'Unknown' tickets into {num_clusters} groups...")
        tfidf_unknown = pipeline.named_steps["tfidf"].transform(unknown_df['clean'])
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        unknown_df['cluster_id'] = kmeans.fit_predict(tfidf_unknown)
        
        # Merge cluster IDs back into the main dataframe
        df.loc[unknown_df.index, 'cluster_id'] = unknown_df['cluster_id']
        print("Clustering complete.")
    else:
        print("Not enough 'Unknown' documents to perform clustering. Skipping clustering.")

    # Drop intermediate columns for a cleaner output
    df.drop(columns=['clean', 'predicted_initial', 'max_score'], inplace=True)

    print(f"Saving predictions to '{output_path}'...")
    df.to_csv(output_path, index=False)
    print("\n--- Prediction and clustering complete. ---")
    print("\nFinal prediction distribution:")
    print(df['predicted_final'].value_counts())


# --- Command-Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IT Service Ticket Classifier and Clustering Tool.")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Operating mode: 'train' or 'predict'")

    # --- Train Mode Arguments ---
    train_parser = subparsers.add_parser("train", help="Train a new classification model.")
    train_parser.add_argument("--data-path", required=True, help="Path to the labeled training CSV file.")
    train_parser.add_argument("--model-path", default="ticket_classifier.joblib", help="Path to save the trained model.")
    train_parser.add_argument("--doc-col", default="Document", help="Name of the document/text column.")
    train_parser.add_argument("--topic-col", default="Topic_group", help="Name of the topic/label column.")
    
    # --- Predict Mode Arguments ---
    predict_parser = subparsers.add_parser("predict", help="Predict topics for new data and cluster unknowns.")
    predict_parser.add_argument("--data-path", required=True, help="Path to the new (unlabeled) data CSV file.")
    predict_parser.add_argument("--model-path", required=True, help="Path to the pre-trained model file.")
    predict_parser.add_argument("--output-path", required=True, help="Path to save the CSV with predictions.")
    predict_parser.add_argument("--doc-col", default="Document", help="Name of the document/text column.")
    predict_parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold to classify as 'Unknown'.")
    predict_parser.add_argument("--num-clusters", type=int, default=8, help="Number of clusters for 'Unknown' tickets.")

    args = parser.parse_args()

    if args.mode == "train":
        train_model(args.data_path, args.model_path, args.doc_col, args.topic_col)
    elif args.mode == "predict":
        predict_and_cluster(args.data_path, args.model_path, args.output_path, args.doc_col, args.threshold, args.num_clusters)