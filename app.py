# app.py

import streamlit as st
import joblib
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy
from spacy import displacy
from sklearn.metrics.pairwise import cosine_similarity

# --- Page Configuration ---
st.set_page_config(page_title="Smart Ticket Assistant", page_icon="💡", layout="wide")

# --- Model & DB Loading ---
MODEL_DIR = 'models'
DB_DIR = 'embedding_db'
CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'ticket_classifier_deep.joblib')
NER_MODEL_PATH = os.path.join(MODEL_DIR, 'ner_model')
EMBEDDINGS_DB_PATH = os.path.join(DB_DIR, 'all_embeddings.npy')
DOCUMENTS_DB_PATH = os.path.join(DB_DIR, 'all_documents.npy')
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

@st.cache_resource
def load_all():
    paths_to_check = [CLASSIFIER_PATH, NER_MODEL_PATH, EMBEDDINGS_DB_PATH, DOCUMENTS_DB_PATH]
    if not all([os.path.exists(p) for p in paths_to_check]):
        return None, None, None, None, None, None
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    classifier_model = joblib.load(CLASSIFIER_PATH)
    ner_model = spacy.load(NER_MODEL_PATH)
    class_names = classifier_model.classes_
    db_embeddings = np.load(EMBEDDINGS_DB_PATH, allow_pickle=True)
    db_documents = np.load(DOCUMENTS_DB_PATH, allow_pickle=True)
    return embedding_model, classifier_model, ner_model, class_names, db_embeddings, db_documents

(embedding_model, classifier, ner_model, class_names, 
 db_embeddings, db_documents) = load_all()

# --- UI ---
st.title("💡 Smart Ticket Assistant")
st.markdown("This tool uses AI to classify tickets, extract key details, and help identify new categories for unknown issues.")

if embedding_model is None:
    st.error(
        "**Models or Database not found!** Please run all setup scripts first.\n\n"
        "1. `python refine_labels.py` (if you haven't)\n"
        "2. `python create_embedding_db.py`\n"
        "3. `python prepare_final_data.py`\n"
        "4. `python train_and_evaluate.py`\n"
        "5. `python train_ner.py`"
    )
else:
    user_input = st.text_area(
        "Enter ticket description here:", 
        height=150, 
        placeholder="e.g., 'My Outlook is crashing and I need a password reset...'"
    )
    
    if st.button("Analyze Ticket"):
        if not user_input.strip():
            st.warning("Please enter a ticket description.")
        else:
            st.markdown("---")
            st.subheader("Analysis Results")
            
            # --- The New, Simpler, and More Powerful Workflow ---
            input_embedding = embedding_model.encode([user_input])
            prediction = classifier.predict(input_embedding)[0]
            
            # --- Check if the prediction is 'Miscellaneous' ---
            if prediction == 'Miscellaneous':
                st.warning("#### Category: Unknown / Miscellaneous")
                st.info("This ticket does not match a known IT category. Analyzing for emerging patterns...")
                
                # Find and display similar tickets to suggest a new category
                st.markdown("##### Potential Emerging Topic: Found Similar Historical Tickets")
                similarities = cosine_similarity(input_embedding, db_embeddings)[0]
                top_5_indices = np.argsort(similarities)[-6:-1][::-1]
                
                for i, index in enumerate(top_5_indices):
                    st.write(f"**Similar Ticket #{i+1} (Similarity: {similarities[index]:.2f}):**")
                    with st.expander("View Ticket"):
                        st.write(db_documents[index])
                st.write("---")
                st.write("An administrator can review these tickets to identify a new category.")

            else:
                # --- This is a known category, proceed as normal ---
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 1. Ticket Category")
                    st.success(f"**Category:** {prediction}")
                    # You can still show probabilities if you want, for context
                    probabilities = classifier.predict_proba(input_embedding)[0]
                    max_proba = probabilities.max()
                    st.info(f"**Confidence:** {max_proba:.0%}")
                
                with col2:
                    st.markdown("#### 2. Extracted Information")
                    doc = ner_model(user_input)
                    if doc.ents:
                        colors = {"SOFTWARE": "#f7a600", "HARDWARE": "#66c2a5", "REQUEST_TYPE": "#8da0cb"}
                        options = {"ents": ["SOFTWARE", "HARDWARE", "REQUEST_TYPE"], "colors": colors}
                        html = displacy.render(doc, style="ent", options=options, jupyter=False)
                        st.markdown(html, unsafe_allow_html=True)
                    else:
                        st.info("No specific entities were detected.")

# --- Sidebar --- remains the same
st.sidebar.header("About the Models")
st.sidebar.markdown(
    """
    **Classifier:**
    - A `LogisticRegression` model trained on `SentenceTransformer` embeddings.
    - It predicts the overall ticket category.

    **Entity Recognizer:**
    - A hybrid `spaCy` model using rules and a statistical model.
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