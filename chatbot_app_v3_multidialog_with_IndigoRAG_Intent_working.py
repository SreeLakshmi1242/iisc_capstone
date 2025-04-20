import os
from pathlib import Path

import streamlit as st
import asyncio
import nest_asyncio
from transformers import logging, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationSummaryBufferMemory
import time
from langchain_community.llms import HuggingFaceHub  # For cloud deployment

# ----------------------------
# App Configuration
# ----------------------------
st.set_page_config(page_title="Chat with Documents", layout="wide")
st.title("🤖 Chat with Your Documents")

# Configure paths (adjust if needed for cloud storage)
FAISS_INDEX_PATH = Path("./faiss_index")
HF_CACHE_DIR = Path("./huggingface_cache")
HF_CACHE_DIR.mkdir(exist_ok=True, parents=True)

os.environ['HF_HOME'] = str(HF_CACHE_DIR)
os.environ['HF_DATASETS_CACHE'] = str(HF_CACHE_DIR / "datasets")
os.environ['TRANSFORMERS_CACHE'] = str(HF_CACHE_DIR)
os.environ['HUGGINGFACE_HUB_CACHE'] = str(HF_CACHE_DIR)

import huggingface_hub.constants
huggingface_hub.constants.HF_HUB_CACHE = str(HF_CACHE_DIR)

# Fix Streamlit event loop
nest_asyncio.apply()
logging.set_verbosity_error()  # Reduce warnings

# Initialize components
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize LLM for Streamlit Community Cloud (using HuggingFaceHub)
llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                     model_kwargs={"temperature": 0.7, "max_new_tokens": 256})

# ----------------------------
# Helper Functions
# ----------------------------
def display_message(msg, show_analysis=False):
    """Helper function to display messages with proper formatting"""
    avatars = {"Customer": "🙋", "ChatAgent": "🤖"}
    colors = {"Customer": "#DCF8C6", "ChatAgent": "#F1F0F0"}

    if msg['role'] == "Customer":
        st.markdown(
            f"""
            <div style='display: flex; gap: 8px; margin-bottom: 10px;'>
                <div style='background-color: {colors['Customer']}; padding: 10px; border-radius: 10px; max-width: 45%; text-align: left;'>
                    <strong>{avatars['Customer']} Customer</strong><br>
                    <span>{msg['content']}</span>
                </div>
            """,
            unsafe_allow_html=True
        )

        if show_analysis and 'sentiment' in msg:
            st.markdown(
                f"""
                <div style='background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-size: 14px; max-width: 35%; min-width: 150px;'>
                    <strong>🧠 Sentiment:</strong> {msg.get('sentiment', '')}<br>
                    <strong>🎯 Intent:</strong> {msg.get('intent', '')} {f"({msg.get('score', '')}%)" if msg.get('score') else ''}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

    elif msg['role'] == "ChatAgent":
        st.markdown(
            f"""
            <div style='display: flex; justify-content: flex-end; margin-bottom: 10px;'>
                <div style='background-color: {colors['ChatAgent']}; padding: 10px; border-radius: 10px; max-width: 60%; text-align: right;'>
                    <strong>{avatars['ChatAgent']} Assistant</strong><br>
                    <span>{msg['content']}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ----------------------------
# Session Initialization
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "display_stage" not in st.session_state:
    st.session_state.display_stage = 0
if "current_message" not in st.session_state:
    st.session_state.current_message = None

# ----------------------------
# Sidebar: Controls and Upload (with PDF support)
# ----------------------------
st.sidebar.button("🪠 Clear Chat", on_click=lambda: st.session_state.update(
    messages=[],
    display_stage=0,
    current_message=None
))

st.sidebar.markdown("---")
st.sidebar.subheader("📄 Upload New Document (TXT / PDF)")
uploaded_file = st.sidebar.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])

doc_path = "sample_docs.txt"
if uploaded_file is not None:
    os.makedirs("temp_uploads", exist_ok=True)
    file_path = f"temp_uploads/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # If FAISS index exists, remove it so we rebuild
    if os.path.exists("faiss_index"):
        import shutil
        shutil.rmtree("faiss_index")

    # Load documents based on file type
    from langchain.document_loaders import TextLoader, PyPDFLoader
    from langchain.text_splitter import CharacterTextSplitter

    if uploaded_file.name.endswith(".txt"):
        loader = TextLoader(file_path)
        documents = loader.load()
    elif uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        # Optional: add source metadata
        for doc in documents:
            doc.metadata["source"] = uploaded_file.name

    # Split documents into chunks
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Embed and save FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local("faiss_index")

    st.sidebar.success("✅ Document processed and vector DB updated.")

# ----------------------------
# Embedding & Retrieval
# ----------------------------
# Load or create vector store
try:
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index directory not found at {FAISS_INDEX_PATH}")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    st.sidebar.success("✅ Loaded existing FAISS index")

except Exception as e:
    st.error(f"Error loading FAISS index: {str(e)}")
    st.info("Will create a new index from sample documents...")

    # Fallback: Create new index from sample documents
    loader = TextLoader("sample_docs.txt")
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(FAISS_INDEX_PATH)
    st.sidebar.info("Created new FAISS index from sample documents")

retriever = vectordb.as_retriever()
memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

# ----------------------------
# NLP Pipelines
# ----------------------------

# Load model with neutral detection (3-class)
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

intent_pipe = pipeline("zero-shot-classification")
intent_labels = ['Billing issues', 'Loyalty Program & Miles', 'Flight booking',
                 'Child and Infant Travel', 'Meal and Dietary Preferences',
                 'Flight rescheduling', 'Lost and found', 'Pet travel policy',
                 'Check-in issues', 'Refund policy',
                 'Emergency Situations and Medical Assistance', 'Seat selection',
                 'Cabin Baggage Restrictions', 'Flight cancellation',
                 'Group Booking Discounts', 'Luggage Rules & Allowances',
                 'Baggage issues']

# ----------------------------
# Main Chat Interface
# ----------------------------
# Display previous messages
for msg in st.session_state.messages:
    display_message(msg, show_analysis=(msg['role'] == 'Customer'))

# Handle new user input
user_input = st.chat_input("Say something...")
if user_input and st.session_state.display_stage == 0:
    # Process sentiment and intent
    sentiment_result = sentiment_pipe(user_input)[0]
    sentiment_label = sentiment_result['label']
    sentiment_emoji = {"POSITIVE": "😄", "NEGATIVE": "😞", "NEUTRAL": "😐"}.get(sentiment_label.upper(), "💬")

    intent_result = intent_pipe(user_input, candidate_labels=intent_labels)
    intent_label = intent_result["labels"][0]
    intent_score = round(intent_result["scores"][0] * 100, 2)

    # Store message and move to stage 1
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

# Handle display stages
if st.session_state.display_stage == 1:
    # Show user message only
    temp_msg = {**st.session_state.current_message}
    temp_msg['sentiment'] = None
    display_message(temp_msg)
    time.sleep(0.5)
    st.session_state.display_stage = 2
    st.rerun()

elif st.session_state.display_stage == 2:
    # Show user message with analysis
    display_message(st.session_state.current_message, show_analysis=True)

    # Generate response
    if st.session_state.current_message["response"] is None:
        with st.spinner("Thinking..."):
            result = qa_chain.run(st.session_state.current_message["content"])
            st.session_state.current_message["response"] = result
    time.sleep(0.5)
    st.session_state.display_stage = 3
    st.rerun()

elif st.session_state.display_stage == 3:
    # Show full conversation
    display_message(st.session_state.current_message, show_analysis=True)

    # Stream assistant response
    response_placeholder = st.empty()
    full_response = ""
    for word in st.session_state.current_message["response"].split():
        full_response += word + " "
        response_placeholder.markdown(
            f"""
            <div style='display: flex; justify-content: flex-end;'>
                <div style='background-color: #F1F0F0; padding: 10px; border-radius: 10px; margin-bottom: 10px; max-width: 60%; text-align: right;'>
                    <strong>🤖 ChatAgent</strong><br>
                    <span>{full_response}▌</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        time.sleep(0.03)

    # Finalize and add to history
    st.session_state.messages.append(st.session_state.current_message)
    st.session_state.messages.append({
        "role": "ChatAgent",
        "content": full_response.strip()
    })
    st.session_state.display_stage = 0
    st.session_state.current_message = None
