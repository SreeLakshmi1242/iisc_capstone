import streamlit as st
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
import os
import time
import tempfile

# ----------------------------
# Streamlit Setup
# ----------------------------
st.set_page_config(page_title="Chat with Your Docs", layout="wide")
st.title("🤖 Chat with Your Documents")

# Hugging Face API Key from Streamlit secrets
hf_api_key = st.secrets["auth_key"]

# ----------------------------
# Embeddings and LLM Setup
# ----------------------------
embedding_model_name = "all-MiniLM-L6-v2"
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=hf_api_key, model_name=embedding_model_name
)

# LLM Pipeline (text generation)
huggingface_pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    use_auth_token=hf_api_key
)
llm = HuggingFacePipeline(pipeline=huggingface_pipe)


# ----------------------------
# Session State Initialization
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "display_stage" not in st.session_state:
    st.session_state.display_stage = 0
if "current_message" not in st.session_state:
    st.session_state.current_message = None

# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.button("🪠 Clear Chat", on_click=lambda: st.session_state.update(
    messages=[], display_stage=0, current_message=None
))

st.sidebar.markdown("---")
st.sidebar.subheader("📄 Upload New Document (TXT / PDF)")
uploaded_file = st.sidebar.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])

FAISS_INDEX_PATH = "faiss_index"

# ----------------------------
# Handle Document Upload
# ----------------------------
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    # Remove old FAISS index if exists
    if os.path.exists(FAISS_INDEX_PATH):
        os.system(f"rm -rf {FAISS_INDEX_PATH}")

    # Load the uploaded document
    if uploaded_file.name.endswith(".txt"):
        loader = TextLoader(file_path)
    elif uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Chunk documents
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Create FAISS index
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(FAISS_INDEX_PATH)
    st.sidebar.success("✅ Document processed and FAISS index created.")

# ----------------------------
# Load FAISS Index
# ----------------------------
try:
    vectordb = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    st.sidebar.success("✅ Loaded existing FAISS index")
except Exception as e:
    st.error(f"Error loading FAISS index: {str(e)}")
    st.info("Creating index from default sample file...")
    loader = TextLoader("sample_docs.txt")
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(FAISS_INDEX_PATH)
    st.sidebar.info("Created FAISS index from sample_docs.txt")

# Setup QA chain
retriever = vectordb.as_retriever()
memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

# ----------------------------
# Display Chat
# ----------------------------
for msg in st.session_state.messages:
    st.markdown(
        f"<div style='background-color: #F1F0F0; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>"
        f"<strong>{msg['role']}</strong><br>{msg['content']}</div>",
        unsafe_allow_html=True
    )

user_input = st.text_input("Say something...")

if user_input and st.session_state.display_stage == 0:
    st.session_state.current_message = {
        "role": "Customer",
        "content": user_input,
        "response": None
    }
    st.session_state.display_stage = 1
    st.rerun()

if st.session_state.display_stage == 1:
    st.markdown(
        f"<div style='background-color: #DCF8C6; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>"
        f"<strong>🙋 Customer</strong><br>{st.session_state.current_message['content']}</div>",
        unsafe_allow_html=True
    )
    time.sleep(0.5)
    st.session_state.display_stage = 2
    st.rerun()

elif st.session_state.display_stage == 2:
    st.markdown(
        f"<div style='background-color: #DCF8C6; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>"
        f"<strong>🙋 Customer</strong><br>{st.session_state.current_message['content']}</div>",
        unsafe_allow_html=True
    )
    with st.spinner("Thinking..."):
        response = qa_chain.run(st.session_state.current_message["content"])
        st.session_state.current_message["response"] = response
    time.sleep(0.5)
    st.session_state.display_stage = 3
    st.rerun()

elif st.session_state.display_stage == 3:
    st.markdown(
        f"<div style='background-color: #DCF8C6; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>"
        f"<strong>🙋 Customer</strong><br>{st.session_state.current_message['content']}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div style='background-color: #F1F0F0; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>"
        f"<strong>🤖 ChatAgent</strong><br>{st.session_state.current_message['response']}</div>",
        unsafe_allow_html=True
    )

    st.session_state.messages.append(st.session_state.current_message)
    st.session_state.messages.append({
        "role": "ChatAgent",
        "content": st.session_state.current_message["response"]
    })
    st.session_state.display_stage = 0
    st.session_state.current_message = None
