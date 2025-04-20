import os
from pathlib import Path

#Added Comment - Pankaj

# 1. Set ALL possible cache locations (new HuggingFace versions need this)
cache_dir = Path("D:/huggingface_cache")
cache_dir.mkdir(exist_ok=True, parents=True)

os.environ['HF_HOME'] = str(cache_dir)
os.environ['HF_DATASETS_CACHE'] = str(cache_dir / "datasets")
os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
os.environ['HUGGINGFACE_HUB_CACHE'] = str(cache_dir)

# 2. Patch the cache before any imports (critical!)
import huggingface_hub.constants
huggingface_hub.constants.HF_HUB_CACHE = str(cache_dir)

# 3. Now import transformers
from transformers import pipeline


import streamlit as st
import asyncio
import nest_asyncio
from transformers import logging
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
#from langchain_ollama import OllamaLLM
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationSummaryBufferMemory
import time
from pathlib import Path

# ----------------------------
# App Configuration
# ----------------------------
st.set_page_config(page_title="Chat with Local LLM", layout="wide")
st.title("🤖 Chat with Your Documents (Locally)")
# Configure paths
FAISS_INDEX_PATH = Path("D:/03-Ollama/Dummy Setup chat LLM bot/faiss_index") # Adjust this path as needed
os.environ["HF_HOME"] = "D:/huggingface_cache"  # Set before loading pipeline
os.environ['OLLAMA_MODELS'] = 'D:/ollama_models'  # Stores Ollama models here

# Fix Streamlit event loop
nest_asyncio.apply()
logging.set_verbosity_error()  # Reduce warnings

# Initialize components
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = Ollama(model="mistral")  # Updated Ollama class

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
        os.system("rm -rf faiss_index")

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
# Model Selector and Setup
# ----------------------------
# In your model selection section:
try:
    model_choice = st.selectbox("Choose LLM model:", ["mistral", "phi"])
    
    # Verify model is available
    import subprocess
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if model_choice not in result.stdout:
        st.warning(f"Model '{model_choice}' not found. Pulling it now...")
        subprocess.run(["ollama", "pull", model_choice], check=True)
    
    llm = Ollama(model=model_choice)
    
except subprocess.CalledProcessError as e:
    st.error(f"Failed to setup Ollama: {e}")
    st.error("Make sure Ollama is installed and running (run 'ollama serve')")
    st.stop()
except Exception as e:
    st.error(f"Error initializing LLM: {e}")
    st.stop()

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

# Example output:
result = sentiment_pipe("Can I bring my pet on board?")
print(result)
# [{'label': 'neutral', 'score': 0.89}]  <-- Now detects neutral!


# Intent pipeline
# Use a model fine-tuned on customer service dialogues
# 2. Initialize the pipeline (cached for performance)


# @st.cache_resource
# def load_intent_pipeline():
#     return pipeline(
#         "text-classification",
#         model="joeddav/xlm-roberta-large-xnli",
#         device=0 if st.session_state.get('use_gpu', False) else -1
#     )


# # 3. Improved intent detection function
# def detect_intent(text):
#     intent_labels = [
#         "flight_delay", "cancellation", "booking_help",
#         "baggage_issue", "special_request", "general_inquiry"
#     ] 
    
#     result = load_intent_pipeline()(text, candidate_labels=intent_labels)
#     return {
#         'intent': result['labels'][0],
#         'confidence': round(result['scores'][0] * 100, 2),
#         'sentiment': analyze_sentiment(text)  # Your existing sentiment function
#     }


intent_pipe = pipeline("zero-shot-classification")
#intent_labels = ["ask_info", "greeting", "travel_help", "cancel_request", "fare_inquiry"]

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

    #analysis = detect_intent(user_input)

    # Store message and move to stage 1
    st.session_state.current_message = {
        "role": "Customer",
        "content": user_input,
        "sentiment": f"{sentiment_label} {sentiment_emoji}",
        "intent": intent_label,
        "score": intent_score,
        #"intent": f"{analysis['intent']} ({analysis['confidence']}%)",
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
