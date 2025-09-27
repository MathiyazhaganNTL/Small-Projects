import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama

# ----------------------------
# CSS & Templates
# ----------------------------
def get_css():
    return """
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        display: flex;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        animation: slideIn 0.3s ease-out;
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .chat-message.user {
        background: linear-gradient(135deg, #2b313e 0%, #3d4454 100%);
        border-left: 4px solid #667eea;
    }
    .chat-message.bot {
        background: linear-gradient(135deg, #475063 0%, #5a6578 100%);
        border-left: 4px solid #764ba2;
    }
    .chat-message .avatar {
        width: 20%;
        display: flex;
        align-items: flex-start;
        justify-content: center;
    }
    .chat-message .avatar img {
        max-width: 78px;
        max-height: 78px;
        border-radius: 50%;
        object-fit: cover;
        border: 3px solid rgba(255,255,255,0.1);
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .chat-message .message {
        width: 80%;
        padding: 0 1.5rem;
        color: #fff;
        line-height: 1.6;
    }
    </style>
    """

def get_bot_template():
    return '''
    <div class="chat-message bot">
        <div class="avatar">
            <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png">
        </div>
        <div class="message">{{MSG}}</div>
    </div>
    '''

def get_user_template():
    return '''
    <div class="chat-message user">
        <div class="avatar">
            <img src="https://i.ibb.co/rdZC7LZ/Photo-logo-1.png">
        </div>
        <div class="message">{{MSG}}</div>
    </div>
    '''

# ----------------------------
# Setup
# ----------------------------
load_dotenv()
PERSIST_DIR = Path("faiss_index")
PERSIST_DIR.mkdir(exist_ok=True)

# ----------------------------
# PDF Processing
# ----------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                text += txt + "\n"
    return text

def get_text_chunks(text, chunk_size=800, overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len
    )
    return splitter.split_text(text)

# ----------------------------
# Embeddings & Vectorstore
# ----------------------------
def get_embeddings_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def build_or_load_vectorstore(text_chunks, persist_name="default"):
    embeddings = get_embeddings_model()
    vs_path = PERSIST_DIR / persist_name
    if vs_path.exists():
        # SAFE: Only use True if you created the FAISS store yourself
        return FAISS.load_local(
            str(vs_path),
            embeddings,
            allow_dangerous_deserialization=True
        )
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    vectorstore.save_local(str(vs_path))
    return vectorstore

# ----------------------------
# LLM (Ollama)
# ----------------------------
def get_llm_model():
    return ChatOllama(model="mistral:7b", temperature=0.2)

def ask_llm(question, docs):
    llm = get_llm_model()
    context = "\n".join([d.page_content for d in docs])
    prompt = f"""You are an AI assistant. Answer only based on the context below:

{context}

Question: {question}
Answer:"""
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)

# ----------------------------
# Streamlit App
# ----------------------------
def main():
    st.set_page_config(page_title="📄 PDF Chat AI", page_icon="🤖", layout="wide")
    st.write(get_css(), unsafe_allow_html=True)
    st.markdown('<div class="main-header"><h1>📄 PDF Chat AI</h1></div>', unsafe_allow_html=True)

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # ---------------- Sidebar
    with st.sidebar:
        pdf_docs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
        persist_name = st.text_input("Project Name", "default")

        if st.button("🚀 Process Documents") and pdf_docs:
            raw_text = get_pdf_text(pdf_docs)
            chunks = get_text_chunks(raw_text)
            st.session_state.vectorstore = build_or_load_vectorstore(chunks, persist_name)
            st.success("✅ Documents processed!")

    # ---------------- Chat
    if st.session_state.vectorstore:
        st.subheader("💬 Ask a Question")
        question = st.text_input("Type your question...", key="question_input")

        if st.button("Ask") and question.strip() != "":
            # Show user message
            st.session_state.chat_history.append(("user", question))
            st.session_state.chat_history.append(("bot", "…typing…"))

            # Synchronous retrieval and LLM generation
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
            docs = retriever.get_relevant_documents(question)
            answer = ask_llm(question, docs)

            # Replace typing placeholder with answer
            st.session_state.chat_history[-1] = ("bot", answer)

    # ---------------- Display chat
    if st.session_state.chat_history:
        for role, msg in st.session_state.chat_history:
            template = get_user_template() if role == "user" else get_bot_template()
            st.write(template.replace("{{MSG}}", msg), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
