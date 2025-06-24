# train_ner.py

import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import random
import os

# Import our training data
from ner_training_data import TRAIN_DATA

def train_ner_model():
    """Trains a custom spaCy NER model."""
    
    # --- Configuration ---
    MODEL_OUTPUT_DIR = "models/ner_model"
    NUM_ITERATIONS = 30 # Number of times to loop over the data

    # 1. Create a blank spaCy model
    nlp = spacy.blank("en")
    print("Created blank 'en' model")

    # 2. Create and configure the NER pipeline component
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # 3. Add the custom entity labels to the NER component
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2]) # Add the label (e.g., "SOFTWARE")

    # 4. Start the training
    print("\n--- Starting Training ---")
    optimizer = nlp.begin_training()

    # Disable other pipes during training for efficiency
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        for itn in range(NUM_ITERATIONS):
            random.shuffle(TRAIN_DATA)
            losses = {}
            
            for text, annotations in TRAIN_DATA:
                # Create spaCy Example objects
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                
                # Update the model with the example
                nlp.update([example], drop=0.35, sgd=optimizer, losses=losses)

            print(f"Iteration {itn + 1}/{NUM_ITERATIONS}, Losses: {losses}")

    # 5. Save the trained model to disk
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    nlp.to_disk(MODEL_OUTPUT_DIR)
    print(f"\n✅ Model saved to '{MODEL_OUTPUT_DIR}'")
    
    # 6. Test the trained model on a sample text
    print("\n--- Testing Model ---")
    test_text = "I need to install outlook on my new dell laptop"
    doc = nlp(test_text)
    print("Entities in:", test_text)
    for ent in doc.ents:
        print(f"-> {ent.text} ({ent.label_})")


if __name__ == "__main__":
    train_ner_model()