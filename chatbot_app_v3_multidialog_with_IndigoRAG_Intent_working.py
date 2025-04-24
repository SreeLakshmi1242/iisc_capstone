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

