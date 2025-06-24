# app.py (Deep Learning + NER + Proba Score Version)

import streamlit as st
import joblib
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy
from spacy import displacy

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Ticket Assistant",
    page_icon="💡",
    layout="wide"
)

# --- Model Loading ---
MODEL_DIR = 'models'
CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'ticket_classifier_deep.joblib')
NER_MODEL_PATH = os.path.join(MODEL_DIR, 'ner_model')
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

@st.cache_resource
def load_models():
    if not all([os.path.exists(p) for p in [CLASSIFIER_PATH, NER_MODEL_PATH]]):
        return None, None, None, None

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    classifier_model = joblib.load(CLASSIFIER_PATH)
    ner_model = spacy.load(NER_MODEL_PATH)
    class_names = classifier_model.classes_
    
    return embedding_model, classifier_model, ner_model, class_names

embedding_model, classifier, ner_model, class_names = load_models()

# --- UI ---
st.title("💡 Smart Ticket Assistant")
st.markdown(
    "This tool uses two AI models: one to **classify the ticket category** and another to **extract key entities**."
)

if embedding_model is None:
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
            col1, col2 = st.columns(2)

            # --- Column 1: Classification with Probability ---
            with col1:
                st.markdown("#### 1. Ticket Category")
                input_embedding = embedding_model.encode([user_input])
                
                # --- CHANGE: Use predict_proba instead of decision_function ---
                # This returns a list of probabilities for each class, e.g., [[0.1, 0.8, 0.1]]
                probabilities = classifier.predict_proba(input_embedding)[0]
                
                # Find the highest probability
                max_proba = probabilities.max()
                
                # --- CHANGE: Adjust the threshold to work with probabilities (0.0 to 1.0) ---
                # A threshold of 0.5 means the model must be at least 50% sure.
                CONFIDENCE_THRESHOLD = 0.5

                if max_proba >= CONFIDENCE_THRESHOLD:
                    prediction_index = probabilities.argmax()
                    prediction = class_names[prediction_index]
                    st.success(f"**Category:** {prediction}")
                    # --- CHANGE: Display the confidence as a percentage ---
                    st.info(f"**Confidence:** {max_proba:.0%}")
                else:
                    st.warning("**Category:** Unclassified")
                    st.info(f"Top category was '{class_names[probabilities.argmax()]}' but confidence was too low ({max_proba:.0%}).")


            # --- Column 2: Entity Extraction (NER) ---
            with col2:
                st.markdown("#### 2. Extracted Information")
                doc = ner_model(user_input)
                
                if doc.ents:
                    html = displacy.render(doc, style="ent", jupyter=False)
                    st.markdown(html, unsafe_allow_html=True)
                    st.markdown("##### Detected Entities:")
                    entity_data = [{"Entity": ent.text, "Type": ent.label_} for ent in doc.ents]
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