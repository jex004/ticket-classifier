# refine_labels.py

import pandas as pd
import os

# --- Configuration ---
# The file you are currently using for training
INPUT_CSV_PATH = 'data/IT_tickets.csv' 

# The new file that will be created with the improved labels
OUTPUT_CSV_PATH = 'data/IT_tickets_refined.csv' 

# The column that contains the labels we want to change
LABEL_COLUMN = 'Topic_group' 

# The new, unified category name
NEW_LABEL = 'Access & Permissions' 

# The old labels that we want to merge
LABELS_TO_MERGE = ['Access', 'Administrative rights']

def merge_categories():
    """
    Loads a CSV, merges specified categories into a new one,
    and saves the result to a new file.
    """
    print(f"--- Starting Label Refinement ---")
    
    # 1. Load the dataset
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        print(f"✅ Successfully loaded '{INPUT_CSV_PATH}' with {len(df)} rows.")
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at '{INPUT_CSV_PATH}'. Please check the path.")
        return

    # Check if the label column exists
    if LABEL_COLUMN not in df.columns:
        print(f"❌ ERROR: The column '{LABEL_COLUMN}' was not found in the CSV file.")
        print(f"Available columns are: {df.columns.tolist()}")
        return

    # 2. Show the category counts *before* the change
    print("\nCategory counts BEFORE merging:")
    print(df[LABEL_COLUMN].value_counts())
    
    # 3. Perform the merge
    # Find all rows where the label is in our list of labels to merge
    rows_to_change_mask = df[LABEL_COLUMN].isin(LABELS_TO_MERGE)
    num_rows_changed = rows_to_change_mask.sum()
    
    if num_rows_changed > 0:
        print(f"\nFound {num_rows_changed} rows to merge into '{NEW_LABEL}'.")
        # Replace the old labels with the new one
        df.loc[rows_to_change_mask, LABEL_COLUMN] = NEW_LABEL
        print("✅ Merge complete.")
    else:
        print(f"\nNo rows found with the labels: {LABELS_TO_MERGE}. No changes made.")

    # 4. Show the category counts *after* the change
    print("\nCategory counts AFTER merging:")
    print(df[LABEL_COLUMN].value_counts())
    
    # 5. Save the refined dataframe to a new CSV file
    try:
        # Ensure the 'data' directory exists
        os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
        df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"\n✅ Successfully saved the refined data to '{OUTPUT_CSV_PATH}'.")
    except Exception as e:
        print(f"\n❌ ERROR: Could not save the file. Reason: {e}")

if __name__ == "__main__":
    merge_categories()