# train_ner.py (IMPROVED and CORRECTED)

import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.pipeline import EntityRuler
import random
import os

# Import our training data
from ner_training_data import TRAIN_DATA

def train_ner_model():
    """Trains a custom spaCy NER model."""
    
    # --- Configuration ---
    MODEL_OUTPUT_DIR = "models/ner_model"
    NUM_ITERATIONS = 30 
    
    # 1. Create a blank spaCy model
    nlp = spacy.blank("en")
    print("Created blank 'en' model")

    # --- Corrected Logic ---
    # Step 2a: Add the statistical 'ner' component to the pipeline FIRST.
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")
    print("Added statistical 'ner' component to the pipeline.")

    # Step 2b: NOW, we can safely add the EntityRuler *before* the 'ner' component.
    print("Creating rule-based EntityRuler...")
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    
    # This is your primary tool for improvement. Add new keywords here.
    patterns = [
        # Software - Notice we can add variations
        {"label": "SOFTWARE", "pattern": "outlook"},
        {"label": "SOFTWARE", "pattern": "sap"},
        {"label": "SOFTWARE", "pattern": "teams"},
        {"label": "SOFTWARE", "pattern": "vpn"},
        {"label": "SOFTWARE", "pattern": "zoom"},
        {"label": "SOFTWARE", "pattern": "visual studio"},
        {"label": "SOFTWARE", "pattern": "oracle"},
        {"label": "SOFTWARE", "pattern": "adobe photoshop"},
        {"label": "SOFTWARE", "pattern": "bit bucket"},
        {"label": "SOFTWARE", "pattern": "confluence"},
        {"label": "SOFTWARE", "pattern": "windows"},
        
        # Hardware - Using LOWER to match "Dell Laptop" or "dell laptop"
        {"label": "HARDWARE", "pattern": [{"LOWER": "dell"}, {"LOWER": "laptop"}]},
        {"label": "HARDWARE", "pattern": "laptop"},
        {"label": "HARDWARE", "pattern": "printer"},
        {"label": "HARDWARE", "pattern": "monitor"},
        {"label": "HARDWARE", "pattern": "keyboard"},
        {"label": "HARDWARE", "pattern": "mouse"},
        {"label": "HARDWARE", "pattern": "webcam"},
        {"label": "HARDWARE", "pattern": "server"},
        {"label": "HARDWARE", "pattern": "phone"},
        {"label": "HARDWARE", "pattern": "wifi"},


        # Request Types - Matching multiple words
        {"label": "REQUEST_TYPE", "pattern": "password reset"},
        {"label": "REQUEST_TYPE", "pattern": "admin rights"},
        {"label": "REQUEST_TYPE", "pattern": "access request"},
        {"label": "REQUEST_TYPE", "pattern": [{"LOWER": "new"}, {"LOWER": "license"}]},
        {"label": "REQUEST_TYPE", "pattern": [{"LOWER": "software"}, {"LOWER": "update"}]},
        {"label": "REQUEST_TYPE", "pattern": [{"LOWER": "new"}, {"LOWER": "purchase"}]},
        {"label": "REQUEST_TYPE", "pattern": "install"}
    ]
    ruler.add_patterns(patterns)
    print(f"Added {len(patterns)} patterns to the ruler.")

    # 3. Add the custom entity labels from TRAIN_DATA to the statistical NER component
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # 4. Start the training
    print("\n--- Starting Training ---")
    optimizer = nlp.begin_training()

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in ["ner", "entity_ruler"]]
    with nlp.disable_pipes(*other_pipes):
        for itn in range(NUM_ITERATIONS):
            random.shuffle(TRAIN_DATA)
            losses = {}
            
            for text, annotations in TRAIN_DATA:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], drop=0.35, sgd=optimizer, losses=losses)

            print(f"Iteration {itn + 1}/{NUM_ITERATIONS}, Losses: {losses}")

    # 5. Save the trained model to disk
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    nlp.to_disk(MODEL_OUTPUT_DIR)
    print(f"\n✅ Model saved to '{MODEL_OUTPUT_DIR}'")
    
    # 6. Test the trained model on a sample text
    print("\n--- Testing Model ---")
    test_text = "I need a password reset for outlook on my new dell laptop"
    doc = nlp(test_text)
    print("Entities in:", test_text)
    for ent in doc.ents:
        print(f"-> '{ent.text}' ({ent.label_})")


if __name__ == "__main__":
    train_ner_model()