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

def display_message(msg, show_analysis=False):
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
if "messages" not in st.session_state:
    st.session_state.messages = []
if "display_stage" not in st.session_state:
    st.session_state.display_stage = 0
if "current_message" not in st.session_state:
    st.session_state.current_message = None

# ----------------------------
# Sidebar: Clear and Upload
# ----------------------------
st.sidebar.button("🪠 Clear Chat", on_click=lambda: st.session_state.update(
    messages=[], display_stage=0, current_message=None
))
