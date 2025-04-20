import streamlit as st
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline  # Updated import from langchain_community
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
import os
import time
import nest_asyncio
import numpy as np

# Custom Embedding Class to Ensure Embeddings are in np.ndarray



# ----------------------------
# Streamlit Setup and App Configuration
# ----------------------------
st.set_page_config(page_title="Chat with Local LLM", layout="wide")
st.title("🤖 Chat with Your Documents (Locally)")

# Configure Hugging Face authentication
hf_api_key = st.secrets["auth_key"]  # Retrieve Hugging Face API key from Streamlit secrets

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=hf_api_key, model_name="mixedbread-ai/mxbai-embed-large-v1"
)
# ----------------------------
# Initialize Hugging Face LLM
# ----------------------------
# Initialize Hugging Face pipeline (e.g., GPT-2 or any Hugging Face model)
huggingface_pipe = pipeline("text-generation", model="gpt2", use_auth_token=hf_api_key)
llm = HuggingFacePipeline(pipeline=huggingface_pipe)  # Use HuggingFacePipeline

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
    if uploaded_file.name.endswith(".txt"):
        loader = TextLoader(file_path)
        documents = loader.load()
    elif uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents = loader.load()

    # Split documents into chunks
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Embed and save FAISS index
    embeddings = CustomHuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Use custom embeddings class
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local("faiss_index")

    st.sidebar.success("✅ Document processed and vector DB updated.")

# ----------------------------
# Embedding & Retrieval (FAISS Index)
# ----------------------------
FAISS_INDEX_PATH = "faiss_index"

try:
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"FAISS index directory not found at {FAISS_INDEX_PATH}")
    
    embeddings = CustomHuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Use custom embeddings class
    vectordb = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    st.sidebar.success("✅ Loaded existing FAISS index")
except Exception as e:
    st.error(f"Error loading FAISS index: {str(e)}")
    st.info("Will create a new index from sample documents...")
    loader = TextLoader("sample_docs.txt")
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    embeddings = CustomHuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Use custom embeddings class
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(FAISS_INDEX_PATH)
    st.sidebar.info("Created new FAISS index from sample documents")

retriever = vectordb.as_retriever()
memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

# ----------------------------
# Main Chat Interface
# ----------------------------
# Display previous messages
for msg in st.session_state.messages:
    st.markdown(
        f"<div style='background-color: #F1F0F0; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>"
        f"<strong>{msg['role']}</strong><br>"
        f"<span>{msg['content']}</span></div>",
        unsafe_allow_html=True
    )

# Handle new user input
user_input = st.text_input("Say something...")

if user_input and st.session_state.display_stage == 0:
    # Store user message and move to stage 1
    st.session_state.current_message = {
        "role": "Customer",
        "content": user_input,
        "response": None
    }
    st.session_state.display_stage = 1
    st.rerun()

# Handle display stages
if st.session_state.display_stage == 1:
    # Show user message only
    temp_msg = {**st.session_state.current_message}
    temp_msg['response'] = None  # Hide response for now
    st.markdown(
        f"<div style='background-color: #DCF8C6; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>"
        f"<strong>🙋 Customer</strong><br>"
        f"<span>{temp_msg['content']}</span></div>",
        unsafe_allow_html=True
    )

    time.sleep(0.5)
    st.session_state.display_stage = 2
    st.rerun()

elif st.session_state.display_stage == 2:
    # Show user message with analysis
    st.markdown(
        f"<div style='background-color: #DCF8C6; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>"
        f"<strong>🙋 Customer</strong><br>"
        f"<span>{st.session_state.current_message['content']}</span></div>",
        unsafe_allow_html=True
    )

    # Generate response
    with st.spinner("Thinking..."):
        result = qa_chain.run(st.session_state.current_message["content"])
        st.session_state.current_message["response"] = result
    
    time.sleep(0.5)
    st.session_state.display_stage = 3
    st.rerun()

elif st.session_state.display_stage == 3:
    # Show user message with response
    st.markdown(
        f"<div style='background-color: #DCF8C6; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>"
        f"<strong>🙋 Customer</strong><br>"
        f"<span>{st.session_state.current_message['content']}</span></div>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<div style='background-color: #F1F0F0; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>"
        f"<strong>🤖 ChatAgent</strong><br>"
        f"<span>{st.session_state.current_message['response']}</span></div>",
        unsafe_allow_html=True
    )

    # Finalize and add to history
    st.session_state.messages.append(st.session_state.current_message)
    st.session_state.messages.append({
        "role": "ChatAgent",
        "content": st.session_state.current_message["response"]
    })
    st.session_state.display_stage = 0
    st.session_state.current_message = None
