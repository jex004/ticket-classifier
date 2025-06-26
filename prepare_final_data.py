# prepare_final_data.py

import pandas as pd
import os
import random

# --- Configuration ---
# Use the refined data file with merged categories as our base
INPUT_CSV_PATH = 'data/IT_tickets_refined.csv' 
OUTPUT_CSV_PATH = 'data/IT_tickets_final_training.csv'
MISC_LABEL = 'Miscellaneous'

def generate_negative_examples(count=500):
    """Generates a list of non-IT related text examples."""
    print(f"Generating {count} new 'Miscellaneous' examples...")
    templates = [
        "the weather is nice today",
        "what time is the meeting tomorrow?",
        "i am going on vacation next week",
        "can you send me the marketing report",
        "let's schedule a call to discuss the project timeline",
        "the quarterly budget is due on friday",
        "i like going outside for a walk",
        "the cafeteria is serving pizza today",
        "remember to submit your expense reports",
        "where is the closest coffee shop",
        "i have a dentist appointment at 3 pm",
        "this is not an IT support ticket",
        "please disregard this message",
        "i am writing to inquire about a sales opportunity",
        "the dog is barking loudly",
        "my favorite color is blue",
        "how do i get to the train station",
        "the financial forecast looks positive",
        "we need to hire a new graphic designer",
        "the new marketing campaign is a success"
    ]
    return [random.choice(templates) for _ in range(count)]

def create_final_dataset():
    """
    Loads the refined data, adds generated negative examples to the
    'Miscellaneous' category, and saves a final training file.
    """
    print("--- Starting Final Data Preparation ---")

    # 1. Load the refined dataset
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        print(f"✅ Loaded '{INPUT_CSV_PATH}' with {len(df)} rows.")
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at '{INPUT_CSV_PATH}'.")
        print("Please run 'python refine_labels.py' first.")
        return

    # 2. Generate new negative examples
    negative_texts = generate_negative_examples()
    negative_df = pd.DataFrame({
        'Document': negative_texts,
        'Topic_group': MISC_LABEL
    })

    # 3. Combine the original data with the new negative examples
    final_df = pd.concat([df, negative_df], ignore_index=True)
    print(f"✅ Combined original data with new examples. Total rows: {len(final_df)}")

    # 4. Show the final category distribution
    print("\nFinal category distribution for training:")
    print(final_df['Topic_group'].value_counts())

    # 5. Save the final training data
    try:
        os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
        final_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"\n✅ Successfully saved the final training data to '{OUTPUT_CSV_PATH}'.")
    except Exception as e:
        print(f"\n❌ ERROR: Could not save the file. Reason: {e}")

if __name__ == "__main__":
    create_final_dataset()