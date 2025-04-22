import os
import time
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline

hf_token = st.secrets["auth_key"]

# App title
st.set_page_config(page_title="Chat with your Documents", layout="wide")
st.title("📄 Chat with Documents - Local LLM + FAISS")

# Sidebar configuration
st.sidebar.header("Configuration")

# Model selection
model_name = st.sidebar.selectbox("Select a model", ["mistralai/Mistral-7B-Instruct-v0.1"])

# FAISS folder path
faiss_folder = st.sidebar.text_input("FAISS Index Folder Path", value="./faiss_index")

# Initialize embedding model
@st.cache_resource
def load_embedding():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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

# Load LLM locally using transformers + HuggingFacePipeline
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.5,
        do_sample=True,
        return_full_text=True
    )

    return HuggingFacePipeline(pipeline=pipe)

llm = load_llm()

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Build Conversational Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True
)


# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "display_stage" not in st.session_state:
    st.session_state.display_stage = 1

if "current_message" not in st.session_state:
    st.session_state.current_message = {"content": "", "response": ""}

# Chat UI
st.subheader("Chat Interface")
user_input = st.chat_input("Ask a question about your documents")

if user_input:
    st.session_state.current_message = {"content": user_input, "response": ""}
    st.session_state.display_stage = 2

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat[0])
    with st.chat_message("assistant"):
        st.markdown(chat[1])

# Handle response
if st.session_state.display_stage == 2:
    with st.chat_message("user"):
        st.markdown(st.session_state.current_message["content"])

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = qa_chain.invoke({"question": st.session_state.current_message["content"]})
                answer = response.get("answer", "No answer generated.")
                st.session_state.current_message["response"] = answer
            except Exception as e:
                st.error(f"Error during response generation: {e}")
                st.session_state.current_message["response"] = ""

        st.markdown(st.session_state.current_message["response"])

    # Store chat
    st.session_state.chat_history.append([
        st.session_state.current_message["content"],
        st.session_state.current_message["response"]
    ])
    time.sleep(0.5)
    st.session_state.display_stage = 3
