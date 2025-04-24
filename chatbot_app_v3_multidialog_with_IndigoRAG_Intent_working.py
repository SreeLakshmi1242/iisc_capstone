import os
import time
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationSummaryBufferMemory
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from pathlib import Path

# Streamlit app setup
st.set_page_config(page_title="Chat with your Documents", layout="wide")
st.title("📄 Chat with Documents - Local LLM + FAISS")

# Hugging Face Token
hf_token = st.secrets["auth_key"]

# Sidebar configuration
st.sidebar.header("Configuration")
model_name = st.sidebar.selectbox("Select a model", [
    "google/flan-t5-base",
    "tiiuae/falcon-7b-instruct",
    "HuggingFaceH4/zephyr-7b-beta"
])
FAISS_INDEX_PATH = st.sidebar.text_input("FAISS Index Folder Path", value="./faiss_index")

# Embeddings setup
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Cache vectorstore loading
@st.cache_resource
def load_vectorstore(path):
    return FAISS.load_local(path, embeddings=embedding_function, allow_dangerous_deserialization=True)

# Load FAISS or create new
if not Path(FAISS_INDEX_PATH).exists():
    st.info("Creating FAISS index from `sample_docs.txt`...")
    loader = TextLoader("sample_docs.txt")
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    vectordb = FAISS.from_documents(chunks, embedding_function)
    vectordb.save_local(FAISS_INDEX_PATH)
else:
    try:
        vectordb = load_vectorstore(FAISS_INDEX_PATH)
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        st.stop()

retriever = vectordb.as_retriever()

# Load LLM pipeline
@st.cache_resource
def load_llm_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        token=hf_token
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, temperature=0.5)

llm_pipeline = load_llm_pipeline()
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Setup QA chain
memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

# Initialize state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "display_stage" not in st.session_state:
    st.session_state.display_stage = 0
if "current_message" not in st.session_state:
    st.session_state.current_message = None

# Sidebar clear
st.sidebar.button("🪠 Clear Chat", on_click=lambda: st.session_state.update(
    messages=[], display_stage=0, current_message=None
))

# Sentiment and intent pipelines
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

# Display chat messages
def display_message(msg, show_analysis=False):
    avatars = {"Customer": "🙋", "ChatAgent": "🤖"}
    colors = {"Customer": "#DCF8C6", "ChatAgent": "#F1F0F0"}

    if msg['role'] == "Customer":
        st.markdown(
            f"""
            <div style='display: flex; gap: 8px; margin-bottom: 10px;'>
                <div style='background-color: {colors['Customer']}; padding: 10px; border-radius: 10px; max-width: 45%; text-align: left;'>
                    <strong>{avatars['Customer']} Customer</strong><br>
                    <span
