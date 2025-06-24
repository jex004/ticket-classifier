# prepare_data.py (Final Robust Version)

"""
This script prepares and combines two different IT support ticket datasets
into a single, unified master dataset for training.

This final version can intelligently handle THREE different CSV formats
from the Kaggle dataset by checking for their unique column names.
"""

import pandas as pd
from langdetect import detect, LangDetectException
import os

# --- Configuration ---
ORIGINAL_DATA_PATH = 'data/all_tickets_processed_improved_v3.csv'
# Define all possible new data files
NEW_DATA_PATH_A = 'data/aa_dataset-tickets-multi-lang-5-2-50-version.csv' # The one we are actually using
NEW_DATA_PATH_B = 'data/support_tickets.csv'
OUTPUT_DIR = 'data'
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, 'master_training_data.csv')

# --- Category Mapping for aa_dataset... (which uses 'type' column) ---
CATEGORY_MAPPING_A = {
    'Email': 'Software',
    'Network': 'Network',
    'Security': 'Access',
    'Software': 'Software',
    'Hardware': 'Hardware',
}

# --- Category Mapping for support_tickets.csv ---
CATEGORY_MAPPING_B = {
    'Hardware & Software': 'Software',
    'Technical Issues': 'Hardware',
    'Product Features': 'Software',
    'Payment & Billing': 'Access',
    'Account & Login': 'Access',
    'General Inquiry': 'Miscellaneous',
    'Product Information': 'Miscellaneous',
    'Installation & Setup': 'Software',
    'Compatibility': 'Software',
}

def is_english(text):
    """Detects if a given text is in English."""
    try:
        return detect(str(text)) == 'en'
    except LangDetectException:
        return False

def prepare_and_combine_data():
    """Main function to process and merge the datasets."""
    print("--- Starting Data Preparation and Combination ---")

    # 1. Load Original Dataset
    try:
        df_original = pd.read_csv(ORIGINAL_DATA_PATH)
        print(f"Loaded {len(df_original)} records from original dataset: {ORIGINAL_DATA_PATH}")
        df_original = df_original[['Document', 'Topic_group']]
    except FileNotFoundError:
        print(f"FATAL: Original data file not found at '{ORIGINAL_DATA_PATH}'. Aborting.")
        return

    # 2. Load New Dataset
    df_new = None
    new_data_file_path = None
    if os.path.exists(NEW_DATA_PATH_A):
        new_data_file_path = NEW_DATA_PATH_A
    elif os.path.exists(NEW_DATA_PATH_B):
        new_data_file_path = NEW_DATA_PATH_B
    
    if new_data_file_path:
        df_new = pd.read_csv(new_data_file_path)
        print(f"Loaded {len(df_new)} records from new dataset: {new_data_file_path}")
    else:
        print(f"FATAL: Could not find a recognizable new data file in the 'data' directory. Aborting.")
        return

    # --- FINAL FIX: Process based on available columns ---
    df_new_processed = pd.DataFrame()

    # Case 1: The file has 'subject' and 'body'
    if 'subject' in df_new.columns and 'body' in df_new.columns:
        print("Processing file with 'subject', 'body', and 'type' columns...")
        df_new_processed['Document'] = df_new['subject'].fillna('') + ' ' + df_new['body'].fillna('')
        df_new_processed['Topic_group'] = df_new['type'].map(CATEGORY_MAPPING_A).fillna('Miscellaneous')
        
    # Case 2: The file has 'text'
    elif 'text' in df_new.columns:
        print("Processing file with 'text' and 'category' columns...")
        df_new_processed = df_new[['text', 'category']].copy()
        df_new_processed.rename(columns={'text': 'Document', 'category': 'Topic_group'}, inplace=True)
        df_new_processed['Topic_group'] = df_new_processed['Topic_group'].map(CATEGORY_MAPPING_B).fillna('Miscellaneous')
    
    else:
        print("FATAL: Could not find recognizable columns in the new dataset.")
        print(f"Expected ('subject'/'body') or ('text'). Found: {df_new.columns.tolist()}")
        return

    # 3. Filter for English Tickets (now that we have a standard 'Document' column)
    print("Filtering new dataset for English-only tickets...")
    english_mask = df_new_processed['Document'].apply(is_english)
    df_new_english = df_new_processed[english_mask].copy()
    print(f"Found {len(df_new_english)} English tickets in the new dataset.")

    print("\nValue counts of new data after mapping:")
    print(df_new_english['Topic_group'].value_counts())

    # 5. Combine Datasets
    print("\nCombining original and new processed data...")
    df_master = pd.concat([df_original, df_new_english], ignore_index=True)
    
    df_master.drop_duplicates(subset=['Document'], inplace=True, keep='first')
    df_master.reset_index(drop=True, inplace=True)
    
    print(f"\nTotal records in the new master dataset: {len(df_master)}")
    print("\nFinal value counts of the combined master dataset:")
    print(df_master['Topic_group'].value_counts())

    # 6. Save the Final Dataset
    df_master.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"\n✅ Master training dataset saved successfully to '{OUTPUT_FILE_PATH}'")
    print("\nYou can now use this file as the input for the 'app_setup.py' script.")

if __name__ == "__main__":
    prepare_and_combine_data()