import os
import time
import pandas
import streamlit as st
from pathlib import Path
from transformers import pipeline
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.llms import HuggingFaceHub
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.schema import(AIMessage,HumanMessage,SystemMessage)
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings


hf_token = st.secrets["auth_key"]
# App title
st.set_page_config(page_title="Chat with your Documents", layout="wide")
st.title("📄 Chat with Documents - Local LLM + FAISS")

# Sidebar configuration
st.sidebar.header("Configuration")

# Model selection
model_name = "HuggingFaceH4/zephyr-7b-beta"


# FAISS folder path
# faiss_folder = st.sidebar.text_input("FAISS Index Folder Path", value="./data")

# Initialize embedding model
@st.cache_resource
def load_embedding(texts):
    return HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"use_auth_token": hf_token,"device":"cpu"}).embed_documents(texts)

# embedding_function = load_embedding()

# Load FAISS vector store
# @st.cache_resource
# def load_vectorstore(path):
#     return FAISS.load_local(path, embeddings=embedding_function, allow_dangerous_deserialization=True)

# try:
#     db = load_vectorstore(faiss_folder)
#     retriever = db.as_retriever()
# except Exception as e:
#     st.error(f"Failed to load FAISS index: {e}")
#     st.stop()

# Load LLM
@st.cache_resource
def load_llm():
    return HuggingFaceHub(repo_id=model_name, model_kwargs={"temperature": 0.5, "max_new_tokens": 512},huggingfacehub_api_token=hf_token)

llm = load_llm()

# Build Conversational Chain
def display_message(msg, show_analysis=False):
    """Helper function to display messages with proper formatting"""
    avatars = {"Customer": "🙋", "ChatAgent": "🤖"}
    colors = {"Customer": "#DCF8C6", "ChatAgent": "#F1F0F0"}
    
    if msg['role'] == "Customer":
        st.markdown(
            f"""
            <div style='display: flex; gap: 8px; margin-bottom: 10px;'>
                <div style='background-color: {colors['Customer']}; padding: 10px; border-radius: 10px; max-width: 45%; text-align: left; color: black;'>
                    <strong>{avatars['Customer']} Customer</strong><br>
                    <span>{msg['content']}</span>
                </div>
            """,
            unsafe_allow_html=True
        )
        
        if show_analysis and 'sentiment' in msg:
            st.markdown(
                f"""
                <div style='background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-size: 14px; max-width: 35%; min-width: 150px; color: black;'>
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
                <div style='background-color: {colors['ChatAgent']}; padding: 10px; border-radius: 10px; max-width: 60%; text-align: right; color: black;'>
                    <strong>{avatars['ChatAgent']} Assistant</strong><br>
                    <span>{msg['content']}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ----------------------------
# State Initialization
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "display_stage" not in st.session_state:
    st.session_state.display_stage = 0
if "current_message" not in st.session_state:
    st.session_state.current_message = None
if "sessionMessages" not in st.session_state:
    st.session_state.sessionMessages = []

# Function to clean assistant's response
def clean_response(response):
    """Clean the assistant's response to remove unnecessary context or prompt rules"""
    # Remove any unwanted content like dates, URLs, and other irrelevant text
    clean_content = ' '.join(response.splitlines())  # Remove newlines
    clean_content = ' '.join([word for word in clean_content.split() if not word.startswith("http")])  # Remove URLs
    # Remove irrelevant sections such as "FAQs", "Terms", "Privacy Policy", etc.
    clean_content = ' '.join([word for word in clean_content.split() if word.lower() not in ['faq', 'terms', 'cookie', 'privacy', 'disclaimer']])
    
    return clean_content.strip()

system_instruction = "The assistant should provide detailed explanations."
# Define the custom prompt template
# template =="""
# You are a helpful assistant. Use the context below to answer the question.
# Only answer using the information provided.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """
data_folder = './skywings'

documents = []

for filename in os.listdir(data_folder):
    if filename.endswith('.txt'):
        file_path = os.path.join(data_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            documents.append(Document(page_content=content, metadata={"source": filename}))  # <- wrap it in Document

# Then pass this 'documents' list
embed_model_1 = HuggingFaceInferenceAPIEmbeddings(
    api_key=hf_token,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
h_vectordb = FAISS.from_documents(documents, embed_model_1)
retriever = h_vectordb.as_retriever(score_threshold = 0.7)

# Create the prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template= """Given the following context and a question, generate an answer based on this context only.
In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

CONTEXT: {context}

QUESTION: {question}

ANSWER:
"""


)
chain_type_kwargs = {"prompt": prompt}

# retriever = db.as_retriever()
memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", return_messages=False)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=h_vectordb.as_retriever(search_type="mmr",search_kwargs={"k": 2, "fetch_k":6} ),
                                                 chain_type="stuff",input_key="query",return_source_documents=True,chain_type_kwargs=chain_type_kwargs)



# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     chain_type="stuff",  # basic stuff chain for one-shot answers
#     chain_type_kwargs={"prompt": condense_question_prompt}
# )


chat_history = []
# ----------------------------
# NLP Pipelines
# ----------------------------
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=-1 # Force CPU # Use Hugging Face token if needed
)

intent_pipe = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",   # very strong zero-shot model
    device=-1,                          # forces CPU / API
)
intent_labels = [
    'Billing issues', 'Loyalty Program & Miles', 'Flight booking',
    'Child and Infant Travel', 'Meal and Dietary Preferences',
    'Flight rescheduling', 'Lost and found', 'Pet travel policy',
    'Check-in issues', 'Refund policy', 'Emergency Situations and Medical Assistance',
    'Seat selection', 'Cabin Baggage Restrictions', 'Flight cancellation',
    'Group Booking Discounts', 'Luggage Rules & Allowances', 'Baggage issues'
]


# ----------------------------
# Main Chat Interface
# ----------------------------
# Display previous messages
for msg in st.session_state.messages:
    display_message(msg, show_analysis=(msg['role'] == 'Customer'))

# Handle new user input
user_input = st.chat_input("Say something...")
if user_input and st.session_state.display_stage == 0:
    st.success("Thanks!")
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
    st.success("T2")
    temp_msg = {**st.session_state.current_message}
    temp_msg['sentiment'] = None
    display_message(temp_msg)
    time.sleep(0.5)
    st.session_state.display_stage = 2
    st.rerun()

elif st.session_state.display_stage == 2:
    st.success("T3")
    display_message(st.session_state.current_message, show_analysis=True)
    if st.session_state.current_message["response"] is None:
        with st.spinner("Thinking..."):
            response =  qa_chain(st.session_state.current_message["content"])
            result = response["result"].split("ANSWER:")[1]

                 # result = qa_chain(st.session_state.current_message["content"]).get('result')
            st.session_state.current_message["response"] = result

            chat_history.append((st.session_state.current_message["content"], result))


             


             

            # Append both messages
            st.session_state.messages.append({
                "role": "Customer",
                "content": st.session_state.current_message["content"],
                "sentiment": st.session_state.current_message["sentiment"],
                "intent": st.session_state.current_message["intent"],
                "score": st.session_state.current_message["score"]
            })
            st.session_state.messages.append({
                "role": "ChatAgent",
                "content": result
            })

            # Reset state
            st.session_state.current_message = None
            st.session_state.display_stage = 0
            st.rerun()
