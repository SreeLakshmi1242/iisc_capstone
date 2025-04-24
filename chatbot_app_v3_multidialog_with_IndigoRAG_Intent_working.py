import os
import time
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory

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
    model_kwargs={"use_auth_token": hf_token})

    
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
    return HuggingFaceHub(repo_id=model_name, model_kwargs={"temperature": 0.5, "max_new_tokens": 512},huggingfacehub_api_token=hf_token)

llm = load_llm()
