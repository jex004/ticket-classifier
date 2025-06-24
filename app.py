# app.py (Deep Learning + NER Version)

import streamlit as st
import joblib
import os
import numpy as np
from sentence_transformers import SentenceTransformer
# --- NEW: Import spaCy and Displacy ---
import spacy
from spacy import displacy

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Ticket Assistant",
    page_icon="💡",
    layout="wide" # Use wide layout for better display
)

# --- Model and Artifact Loading ---
MODEL_DIR = 'models'
CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'ticket_classifier_deep.joblib')
NER_MODEL_PATH = os.path.join(MODEL_DIR, 'ner_model') # Path to our custom spaCy model
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'


@st.cache_resource
def load_models():
    """Loads all models: sentence transformer, classifier, and NER."""
    # Check if all required model paths exist
    if not os.path.exists(CLASSIFIER_PATH) or not os.path.exists(NER_MODEL_PATH):
        return None, None, None, None

    # Load the powerful sentence embedding model
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # Load our trained logistic regression classifier
    classifier_model = joblib.load(CLASSIFIER_PATH)
    
    # Load our custom trained spaCy NER model
    ner_model = spacy.load(NER_MODEL_PATH)
    
    class_names = classifier_model.classes_
    
    return embedding_model, classifier_model, ner_model, class_names

# Load all models
embedding_model, classifier, ner_model, class_names = load_models()

# --- Application UI ---
st.title("💡 Smart IT Ticket Assistant")
st.markdown(
    "This tool uses two AI models: one to **classify the ticket category** and another to **extract key entities** like software and hardware."
)

if embedding_model is None or classifier is None or ner_model is None:
    st.error(
        "**Models not found!** Please run the training scripts first.\n\n"
        "1. Run: `python train_and_evaluate.py`\n"
        "2. Run: `python train_ner.py`"
    )
else:
    user_input = st.text_area(
        "Enter ticket description here:", 
        height=150, 
        placeholder="e.g., 'My Outlook is crashing and I need a password reset for SAP on my Dell laptop.'"
    )
    
    if st.button("Analyze Ticket"):
        if not user_input.strip():
            st.warning("Please enter a ticket description.")
        else:
            st.markdown("---")
            st.subheader("Analysis Results")

            # --- Create two columns for a cleaner layout ---
            col1, col2 = st.columns(2)

            # --- Column 1: Classification ---
            with col1:
                st.markdown("#### 1. Ticket Category")
                # 1. Create embedding
                input_embedding = embedding_model.encode([user_input])
                # 2. Get decision scores
                scores = classifier.decision_function(input_embedding)[0]
                max_score = scores.max()
                CONFIDENCE_THRESHOLD = 0.0

                if max_score >= CONFIDENCE_THRESHOLD:
                    prediction_index = scores.argmax()
                    prediction = class_names[prediction_index]
                    st.success(f"**Category:** {prediction}")
                    st.info(f"**Confidence Score:** {max_score:.2f}")
                else:
                    st.warning("**Category:** Unclassified")

            # --- Column 2: Entity Extraction (NER) ---
            with col2:
                st.markdown("#### 2. Extracted Information")
                # Run the NER model on the user input
                doc = ner_model(user_input)
                
                if doc.ents:
                    # Use Displacy to render the entities
                    html = displacy.render(doc, style="ent", jupyter=False)
                    st.markdown(html, unsafe_allow_html=True)
                    
                    st.markdown("##### Detected Entities:")
                    # Display entities in a more structured way
                    entity_data = [
                        {"Entity": ent.text, "Type": ent.label_}
                        for ent in doc.ents
                    ]
                    st.table(entity_data)

                else:
                    st.info("No specific entities (like software or hardware) were detected.")

# --- Sidebar ---
st.sidebar.header("About the Models")
st.sidebar.markdown(
    """
    **Classifier:**
    - A `LogisticRegression` model trained on `SentenceTransformer` embeddings.
    - It predicts the overall ticket category.

    **Entity Recognizer:**
    - A custom `spaCy` model.
    - It is trained to find and label specific keywords in the text.
    """
)

if class_names is not None:
    st.sidebar.header("Known Categories")
    class_list_markdown = "\n".join([f"- {c}" for c in sorted(class_names)])
    st.sidebar.markdown(class_list_markdown)

if ner_model is not None:
    st.sidebar.header("Known Entities")
    entity_list_markdown = "\n".join([f"- {label}" for label in sorted(ner_model.get_pipe('ner').labels)])
    st.sidebar.markdown(entity_list_markdown)   