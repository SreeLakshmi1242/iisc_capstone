import os
import time
import streamlit as st
from transformers import pipeline
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.memory import ConversationSummaryBufferMemory
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from pathlib import Path

# App Configuration
st.set_page_config(page_title="Chat with your Documents", layout="wide")
st.title("📄 Chat with Documents - Local LLM + FAISS")

# Hugging Face Token
hf_token = st.secrets["auth_key"]

# Sidebar Configuration
st.sidebar.header("Configuration")
model_name = st.sidebar.selectbox("Select a model", ["HuggingFaceH4/zephyr-7b-beta"])
FAISS_INDEX_PATH = st.sidebar.text_input("FAISS Index Folder Path", value="./faiss_index")


# Load FAISS vector store
@st.cache_resource
def load_vectorstore(path):
    try:
        return FAISS.load_local(path, embeddings=embedding_function, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Failed to load FAISS index: {e}")
        st.stop()

# Try loading FAISS vector store
try:
    db = load_vectorstore(FAISS_INDEX_PATH)
    retriever = db.as_retriever()
except Exception as e:
    st.error(f"Failed to load FAISS index: {e}")
    st.stop()

# Load LLM
@st.cache_resource
def load_llm():
    return HuggingFaceHub(
        repo_id=model_name,
        model_kwargs={"temperature": 0.5, "max_new_tokens": 512},
        huggingfacehub_api_token=hf_token
    )

