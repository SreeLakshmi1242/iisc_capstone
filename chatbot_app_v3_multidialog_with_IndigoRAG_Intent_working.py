import os
import time
import streamlit as st
from pathlib import Path
from transformers import pipeline
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.llms import HuggingFaceHub
from langchain.memory import ConversationSummaryBufferMemory

hf_token = st.secrets["auth_key"]
# App title
st.set_page_config(page_title="Chat with your Documents", layout="wide")
st.title("📄 Chat with Documents - Local LLM + FAISS")

# Sidebar configuration
st.sidebar.header("Configuration")

# Model selection
model_name = st.sidebar.selectbox("Select a model", [
    "HuggingFaceH4/zephyr-7b-beta"
])

# FAISS folder path
faiss_folder = st.sidebar.text_input("FAISS Index Folder Path", value="./faiss_index")

# Initialize embedding model
@st.cache_resource
def load_embedding():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"use_auth_token": hf_token, "device": "cpu"}
    )
    
embedding_function = load_embedding()

# Load FAISS vector store
@st.cache_resource
def load_vectorstore(path):
    return FAISS.load_local(path, embeddings=embedding_function, allow_dangerous_deserialization=True)

try:
    db = load_vectorstore(faiss_folder)
    retriever = db.as_retriever()
except Exception as e:
    st.error(f"Failed to load FAISS index: {e}")
    st.stop()

# Load LLM
@st.cache_resource
def load_llm():
    return HuggingFaceHub(repo_id=model_name, model_kwargs={"temperature": 0.5, "max_new_tokens": 512}, huggingfacehub_api_token=hf_token)

llm = load_llm()

# Build Conversational Chain
def display_generated_text(msg):
    """Helper function to display only the generated text without avatars or extra layout"""
    st.markdown(f"{msg['content']}")

# State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "display_stage" not in st.session_state:
    st.session_state.display_stage = 0
if "current_message" not in st.session_state:
    st.session_state.current_message = None

# ----------------------------
# NLP Pipelines
# ----------------------------
sentiment_pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

intent_pipe = pipeline("zero-shot-classification")
intent_labels = [
    'Billing issues', 'Loyalty Program & Miles', 'Flight booking',
    'Child and Infant Travel', 'Meal and Dietary Preferences',
    'Flight rescheduling', 'Lost and found', 'Pet travel policy',
    'Check-in issues', 'Refund policy', 'Emergency Situations and Medical Assistance',
    'Seat selection', 'Cabin Baggage Restrictions', 'Flight cancellation',
    'Group Booking Discounts', 'Luggage Rules & Allowances', 'Baggage issues'
]

# ----------------------------
# Main Chat Interface
# ----------------------------
# Display previous messages
for msg in st.session_state.messages:
    # Show only the "ChatAgent" (assistant) message without avatar and layout
    if msg['role'] == "ChatAgent" and msg['content'] is not None:
        display_generated_text(msg)

# Handle new user input
user_input = st.chat_input("Say something...")
if user_input and st.session_state.display_stage == 0:
    st.success("Thanks!")
    sentiment_result = sentiment_pipe(user_input)[0]
    sentiment_label = sentiment_result['label']
    sentiment_emoji = {"POSITIVE": "😄", "NEGATIVE": "😞", "NEUTRAL": "😐"}.get(sentiment_label.upper(), "💬")

    intent_result = intent_pipe(user_input, candidate_labels=intent_labels)
    intent_label = intent_result["labels"][0]
    intent_score = round(intent_result["scores"][0] * 100, 2)

    st.session_state.current_message = {
        "role": "Customer",
        "content": user_input,
        "sentiment": f"{sentiment_label} {sentiment_emoji}",
        "intent": intent_label,
        "score": intent_score,
        "response": None
    }
    st.session_state.display_stage = 1
    st.rerun()

elif st.session_state.display_stage == 1:
    st.success("T2")
    temp_msg = {**st.session_state.current_message}
    temp_msg['sentiment'] = None
    display_generated_text(temp_msg)
    time.sleep(0.5)
    st.session_state.display_stage = 2
    st.rerun()

elif st.session_state.display_stage == 2:
    st.success("T3")
    display_generated_text(st.session_state.current_message)
    if st.session_state.current_message["response"] is None:
        with st.spinner("Thinking..."):
            result = qa_chain.run(st.session_state.current_message["content"])
            st.session_state.current_message["response"] = result

            # Append both messages
            st.session_state.messages.append({
                "role": "Customer",
                "content": st.session_state.current_message["content"],
                "sentiment": st.session_state.current_message["sentiment"],
                "intent": st.session_state.current_message["intent"],
                "score": st.session_state.current_message["score"]
            })
            st.session_state.messages.append({
                "role": "ChatAgent",
                "content": result
            })

            # Reset state
            st.session_state.current_message = None
            st.session_state.display_stage = 0
            st.rerun()
