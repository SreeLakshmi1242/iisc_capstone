import os
import time
import streamlit as st
from pathlib import Path
from transformers import pipeline
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
    model_kwargs={"use_auth_token": hf_token,"device":"cpu"})

    
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

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Build Conversational Chain
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


def load_vectorstore():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(faiss_folder, embeddings=embedding, allow_dangerous_deserialization=True)
    return vectorstore

# Load on app start
with st.spinner("Loading FAISS vector store..."):
    vectorstore = load_vectorstore()
    st.success("FAISS index loaded successfully!")

# Search box
query = st.text_input("Ask a question or search your documents:")

# Perform similarity search
if query:
    results = vectorstore.similarity_search(query, k=3)
    st.write("Top matches:")
    for i, res in enumerate(results, 1):
        st.markdown(f"**Result {i}:** {res.page_content}")

 # ----------------------------
# State Initialization
# ----------------------------
# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "display_stage" not in st.session_state:
#     st.session_state.display_stage = 0
# if "current_message" not in st.session_state:
#     st.session_state.current_message = None

# # ----------------------------
# # Sidebar: Clear and Upload
# # ----------------------------
# st.sidebar.button("🪠 Clear Chat", on_click=lambda: st.session_state.update(
#     messages=[], display_stage=0, current_message=None
# ))


# st.sidebar.markdown("---")
# st.sidebar.subheader("📄 Upload New Document (TXT / PDF)")
# uploaded_file = st.sidebar.text_input("FAISS Index Folder Path", value="./faiss_index")


# if uploaded_file:
#     os.makedirs("temp_uploads", exist_ok=True)
#     file_path = f"temp_uploads/{uploaded_file.name}"
#     with open(file_path, "wb") as f:
#         f.write(uploaded_file.read())
    
#     if os.path.exists("faiss_index"):
#         os.system("rm -rf faiss_index")

#     if uploaded_file.name.endswith(".txt"):
#         loader = TextLoader(file_path)
#     else:
#         loader = PyPDFLoader(file_path)

#     documents = loader.load()
#     for doc in documents:
#         doc.metadata["source"] = uploaded_file.name

#     splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     chunks = splitter.split_documents(documents)

#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     vectordb = FAISS.from_documents(chunks, embeddings)
#     vectordb.save_local("faiss_index")

#     st.sidebar.success("✅ Document processed and vector DB updated.")
#     st.rerun()

# ----------------------------
# Load or create FAISS index
# if not Path(FAISS_INDEX_PATH).exists():
#     st.info("Creating a new FAISS index from sample_docs.txt")
#     loader = TextLoader("sample_docs.txt")
#     documents = loader.load()
#     splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     chunks = splitter.split_documents(documents)
#     vectordb = FAISS.from_documents(chunks, embedding_function)
#     vectordb.save_local(FAISS_INDEX_PATH)
# else:
# try:
#     vectordb = load_vectorstore(faiss_folder)  # Make sure this is defined
# except Exception as e:
#     st.error(f"Error loading FAISS vector store: {e}")
#     st.stop()

# # ----------------------------
# # NLP Pipelines
# # ----------------------------
# sentiment_pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# intent_pipe = pipeline("zero-shot-classification")
# intent_labels = [
#     'Billing issues', 'Loyalty Program & Miles', 'Flight booking',
#     'Child and Infant Travel', 'Meal and Dietary Preferences',
#     'Flight rescheduling', 'Lost and found', 'Pet travel policy',
#     'Check-in issues', 'Refund policy', 'Emergency Situations and Medical Assistance',
#     'Seat selection', 'Cabin Baggage Restrictions', 'Flight cancellation',
#     'Group Booking Discounts', 'Luggage Rules & Allowances', 'Baggage issues'
# ]

# # ----------------------------
# # Main Chat Interface
# # ----------------------------
# # Display previous messages
# for msg in st.session_state.messages:
#     display_message(msg, show_analysis=(msg['role'] == 'Customer'))

# # Handle new user input
# user_input = st.chat_input("Say something...")
# if user_input and st.session_state.display_stage == 0:
#     sentiment_result = sentiment_pipe(user_input)[0]
#     sentiment_label = sentiment_result['label']
#     sentiment_emoji = {"POSITIVE": "😄", "NEGATIVE": "😞", "NEUTRAL": "😐"}.get(sentiment_label.upper(), "💬")

#     intent_result = intent_pipe(user_input, candidate_labels=intent_labels)
#     intent_label = intent_result["labels"][0]
#     intent_score = round(intent_result["scores"][0] * 100, 2)

#     st.session_state.current_message = {
#         "role": "Customer",
#         "content": user_input,
#         "sentiment": f"{sentiment_label} {sentiment_emoji}",
#         "intent": intent_label,
#         "score": intent_score,
#         "response": None
#     }
#     st.session_state.display_stage = 1
#     st.rerun()

# elif st.session_state.display_stage == 1:
#     temp_msg = {**st.session_state.current_message}
#     temp_msg['sentiment'] = None
#     display_message(temp_msg)
#     time.sleep(0.5)
#     st.session_state.display_stage = 2
#     st.rerun()

# elif st.session_state.display_stage == 2:
#     display_message(st.session_state.current_message, show_analysis=True)
#     if st.session_state.current_message["response"] is None:
#         with st.spinner("Thinking..."):
#             result = qa_chain.run(st.session_state.current_message["content"])
#             st.session_state.current_message["response"] = result
