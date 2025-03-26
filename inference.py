import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import http.client
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import string
from ipc import *  # Import IPC prediction function

# Set Streamlit Theme
st.set_page_config(page_title="LEGAL SUMM", page_icon="⚖️", layout="centered")

# Custom CSS for pastel-colored legal theme
# st.markdown("""
#     <style>
#         body {
#             background-color: #f8f9fa;
#             color: #333;
#         }
#         .stButton > button {
#             background-color: #6c757d;
#             color: white;
#             border-radius: 8px;
#             padding: 10px 20px;
#         }
#         .stButton > button:hover {
#             background-color: #495057;
#         }
#         .stTextArea, .stTextInput {
#             border-radius: 8px;
#             padding: 10px;
#         }
#     </style>
# """, unsafe_allow_html=True)

st.markdown(
    """
    <style>
        body {background-color: #dff6ff;}
        .stButton > button { width: 100%; padding: 12px; margin-bottom: 10px; border-radius: 8px; font-size: 18px; }
        .stTitle { text-align: center; color: #3d5a80; }
        .stHeader { text-align: center; color: #293241; }
    </style>
    """,
    unsafe_allow_html=True
)


# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load summarization model
tokenizer = AutoTokenizer.from_pretrained("Falconsai/text_summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("Falconsai/text_summarization").to(device)

# Load trained ML model (Ensure these files exist and are pre-trained)
x_train_df = pd.read_csv("X_train.csv")
y_train_df = pd.read_csv("y_train.csv")

language_map = {
    "Bengali": "bn", "Gujarati": "gu", "Hindi": "hi", "Kannada": "kn", "Marathi": "mr",
    "Odia": "or", "Punjabi": "pa", "Sanskrit": "sa", "Tamil": "ta", "Telugu": "te",
    "Urdu": "ur", "Malayalam": "ml", "Bhojpuri": "bho", "Sindhi": "sd"
}

trained_model = make_pipeline(TfidfVectorizer(stop_words='english'), LogisticRegression())
trained_model.fit(x_train_df['Facts'], y_train_df['winner_index'])

def preprocess_text(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation))

def predict_winner(party_a, party_b, facts):
    processed_facts = preprocess_text(facts)
    input_text = f"{party_a} {party_b} {processed_facts}"
    probabilities = trained_model.predict_proba([input_text])[0]
    return party_a if probabilities[0] > probabilities[1] else party_b

# Streamlit UI
st.title("⚖️ LEGAL SUMM")

if "page" not in st.session_state:
    st.session_state.page = "main"

if st.session_state.page == "main":
    st.header("Choose a Feature")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📄 Summarization & Translation"):
            st.session_state.page = "summary"
    with col2:
        if st.button("⚖️ Winner Prediction"):
            st.session_state.page = "winner"
    with col3:
        if st.button("📜 IPC Section Prediction"):
            st.session_state.page = "ipc"

elif st.session_state.page == "summary":
    st.header("📄 Text Summarizer & Translator")
    text_input = st.text_area("Enter text to summarize:")
    selected_language = st.selectbox("Select translation language:", list(language_map.keys()))
    if st.button("Summarize"):
        prepared_text = "summarize: " + text_input
        tokenized_text = tokenizer.encode(prepared_text, return_tensors="pt").to(device)
        summary_ids = model.generate(tokenized_text, num_beams=4, min_length=50, max_length=256, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        st.subheader("Summarized Text:")
        st.write(summary)
        target_lang_code = language_map[selected_language]
        conn = http.client.HTTPSConnection("openl-translate.p.rapidapi.com")
        payload = json.dumps({"target_lang": target_lang_code, "text": summary})
        headers = {
            'x-rapidapi-key': "7ed2261e29msh722bffd5ba42056p13c263jsn745de4a3d57c",
            'x-rapidapi-host': "openl-translate.p.rapidapi.com",
            'Content-Type': "application/json"
        }
        conn.request("POST", "/translate", payload, headers)
        res = conn.getresponse()
        data = res.read()
        st.subheader("Translated Summary:")
        st.write(data.decode("utf-8"))
    if st.button("🔙 Back"):
        st.session_state.page = "main"

elif st.session_state.page == "winner":
    st.header("⚖️ Winner Probability Prediction")
    party_a = st.text_input("Enter Party A:")
    party_b = st.text_input("Enter Party B:")
    facts = st.text_area("Enter Case Facts:")
    if st.button("Predict Winner"):
        if party_a and party_b and facts:
            winner = predict_winner(party_a, party_b, facts)
            st.subheader("Predicted Winning Party:")
            st.write(winner)
        else:
            st.warning("Please fill all fields.")
    if st.button("🔙 Back"):
        st.session_state.page = "main"

elif st.session_state.page == "ipc":
    st.header("📜 IPC Section Prediction")
    case_facts = st.text_area("Enter case details:")
    if st.button("Predict IPC Sections"):
        if case_facts:
            sections, probabilities = predict_section_and_punishment(case_facts)
            st.subheader("Predicted IPC Sections:")
            for sec, prob in zip(sections, probabilities):
                st.write(f"📌 IPC Section: {sec}")
        else:
            st.warning("Please enter case details.")
    if st.button("🔙 Back"):
        st.session_state.page = "main"
