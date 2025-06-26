# create_embedding_db.py

import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# --- Configuration ---
# Use the same refined data file that your classifier trains on
DATA_FILE_PATH = 'data/IT_tickets_refined.csv'
DOCUMENT_COL = 'Document'

# Where to save the final database files
DB_DIR = 'embedding_db'
EMBEDDINGS_PATH = os.path.join(DB_DIR, 'all_embeddings.npy')
DOCUMENTS_PATH = os.path.join(DB_DIR, 'all_documents.npy')

# The same model used in your main app
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

def create_database():
    """
    Encodes all documents in the CSV and saves the embeddings and
    the documents themselves to disk for later retrieval.
    """
    print("--- Starting Embedding Database Creation ---")

    # 1. Load the dataset
    try:
        df = pd.read_csv(DATA_FILE_PATH)
        df.dropna(subset=[DOCUMENT_COL], inplace=True)
        print(f"✅ Loaded {len(df)} documents from '{DATA_FILE_PATH}'.")
    except FileNotFoundError:
        print(f"❌ ERROR: Data file not found at '{DATA_FILE_PATH}'.")
        return

    # 2. Load the Sentence Transformer model
    print(f"Loading sentence transformer model '{EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # 3. Encode all documents
    print("Encoding all documents... (This may take a while)")
    documents = df[DOCUMENT_COL].tolist()
    embeddings = model.encode(documents, show_progress_bar=True)

    # 4. Save the embeddings and the documents
    os.makedirs(DB_DIR, exist_ok=True)
    
    # Save embeddings as a NumPy array
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"✅ Saved {len(embeddings)} embeddings to '{EMBEDDINGS_PATH}'")

    # Save the corresponding documents for easy lookup
    np.save(DOCUMENTS_PATH, np.array(documents, dtype=object))
    print(f"✅ Saved {len(documents)} documents to '{DOCUMENTS_PATH}'")

    print("\n--- Database Creation Complete ---")

if __name__ == "__main__":
    create_database()