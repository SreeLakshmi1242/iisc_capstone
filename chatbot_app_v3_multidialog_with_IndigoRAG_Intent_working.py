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

# Display previous messages
for msg in st.session_state.messages:
    display_message(msg, show_analysis=(msg['role'] == 'Customer'))

# Handle new input
user_input = st.chat_input("Say something...")
if user_input and st.session_state.display_stage == 0:
    sentiment_result = sentiment_pipe(user_input)[0]
    sentiment_label = sentiment_result['label']
    sentiment_emoji = {"POSITIVE": "😄", "NEGATIVE": "😞", "NEUTRAL": "😐"}.get(sentiment_label.upper(), "💬")

    intent_result = intent_pipe(user_input, candidate_labels=intent_labels)
    intent_label = intent_result["labels"][0]
    intent_score = round(intent_result["scores"][0] * 100, 2)

    st.session_state.current_message = {
        "role": "Customer",
        "content": user_input,
        "sentiment": f"{sentiment_label} {sentiment_emoji}",
        "intent": intent_label,
        "score": intent_score,
        "response": None
    }
    st.session_state.display_stage = 1
    st.rerun()

elif st.session_state.display_stage == 1:
    temp_msg = {**st.session_state.current_message}
    temp_msg['sentiment'] = None
    display_message(temp_msg)
    time.sleep(0.5)
    st.session_state.display_stage = 2
    st.rerun()

elif st.session_state.display_stage == 2:
    display_message(st.session_state.current_message, show_analysis=True)
    if st.session_state.current_message["response"] is None:
        with st.spinner("Thinking..."):
            result = qa_chain.run(st.session_state.current_message["content"])
            st.session_state.current_message["response"] = result
            st.session_state.messages.append(st.session_state.current_message)
            st.session_state.messages.append({"role": "ChatAgent", "content": result})
        st.session_state.display_stage = 0
        st.rerun()
