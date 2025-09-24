import os
import requests
import streamlit as st
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from PyPDF2 import PdfReader
import time
import threading
import queue
import datetime
import uuid
import json
import re

# ---------------------- ENV ----------------------
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# ---------------------- Persistent ChromaDB ----------------------
# ✅ Use PersistentClient instead of deprecated Client(Settings(...))
chroma_client = chromadb.PersistentClient(path="./chroma_db", settings=Settings())

collection = chroma_client.get_or_create_collection(
    name="user_docs",
    metadata={"hnsw:space": "cosine"}
)

# ---------------------- DB Worker (flush every second) ----------------------
if "db_queue" not in st.session_state:
    st.session_state["db_queue"] = queue.Queue()

if "db_worker_started" not in st.session_state:
    st.session_state["db_worker_started"] = False

def db_worker_loop(q, client, coll, flush_interval=1, max_batch=256):
    while True:
        time.sleep(flush_interval)
        batch = []
        try:
            while len(batch) < max_batch:
                item = q.get_nowait()
                batch.append(item)
        except queue.Empty:
            pass

        if not batch:
            continue

        try:
            documents = [it["document"] for it in batch]
            embeddings = [it["embedding"] for it in batch]
            ids = [it["id"] for it in batch]
            metadatas = [it.get("metadata", {}) for it in batch]

            coll.add(
                documents=documents,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas,
            )

            # ✅ no need for client.persist() — PersistentClient handles it
            st.session_state.setdefault("debug_logs", []).append(
                f"✅ Persisted {len(batch)} items at {datetime.datetime.utcnow().isoformat()}Z"
            )
        except Exception as e:
            st.session_state.setdefault("debug_logs", []).append(f"❌ DB persist error: {e}")
            for it in batch:  # re-enqueue
                try:
                    q.put_nowait(it)
                except Exception:
                    pass

if not st.session_state["db_worker_started"]:
    t = threading.Thread(
        target=db_worker_loop,
        args=(st.session_state["db_queue"], chroma_client, collection),
        daemon=True,
    )
    t.start()
    st.session_state["db_worker_started"] = True

# ---------------------- Helpers ----------------------
def request_with_retry(url, payload, retries=5, timeout=60, stream=False):
    for attempt in range(retries):
        try:
            response = requests.post(url, json=payload, timeout=timeout, stream=stream)
            response.raise_for_status()
            return response
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                st.warning(f"⚠️ Request failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"❌ Request failed after {retries} attempts: {e}")

def get_embedding(text: str):
    try:
        response = request_with_retry(
            "http://localhost:11434/api/embeddings",
            {"model": "nomic-embed-text", "prompt": text, "stream": False},
            timeout=60,
        )
        data = response.json()
        return data.get("embedding", None)
    except Exception as e:
        st.error(f"⚠️ Embedding failed: {e}")
        return None

def chat_with_ollama(context, query, model_name):
    try:
        try:
            check_response = requests.get("http://localhost:11434/api/version", timeout=5)
            check_response.raise_for_status()
        except Exception as e:
            yield f"⚠️ Ollama not available: {e}"
            return

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "Answer concisely using the context provided."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"},
            ],
            "stream": True,
        }

        response = request_with_retry(
            "http://localhost:11434/api/chat", payload, timeout=300, stream=True
        )

        answer = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = line.decode("utf-8")
                    json_data = json.loads(data)
                    if json_data.get("message", {}).get("content"):
                        part = json_data["message"]["content"]
                        answer += part
                        yield answer
                except Exception:
                    if '"content":' in data:
                        content_match = re.search(r'"content":\s*"([^"]*)"', data)
                        if content_match:
                            part = content_match.group(1)
                            answer += part
                            yield answer
        return
    except Exception as e:
        yield f"⚠️ Chat failed: {e}"

def fetch_news(topic, api_key):
    try:
        url = f"https://newsapi.org/v2/everything?q={topic}&sortBy=publishedAt&apiKey={api_key}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        if "articles" in data:
            return [
                f"{a['title']} - {a['description']} ({a['publishedAt']})"
                for a in data["articles"]
                if a.get("title") and a.get("description")
            ]
        return []
    except Exception as e:
        st.error(f"⚠️ Failed to fetch news: {e}")
        return []

def split_text(text, chunk_size=1000, overlap=100):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ---------------------- UI ----------------------
st.set_page_config(page_title="Real-Time RAG with Ollama", layout="wide")
st.title("📡 Real-Time RAG with Ollama")

model_choice = st.selectbox("Choose Ollama Model:", ["llama2:7b", "mistral:7b"])
mode = st.radio("Choose Mode:", ["📂 Upload Docs", "📰 News Search"])
st.write("🔑 News API Key loaded:", NEWS_API_KEY is not None)

# ---------------------- Upload Docs ----------------------
if mode == "📂 Upload Docs":
    if 'debug_logs' not in st.session_state:
        st.session_state['debug_logs'] = []
    with st.sidebar:
        debug_mode = st.checkbox("Show Debug Info", value=False)
        if debug_mode and st.button("Clear Debug Logs"):
            st.session_state['debug_logs'] = []
        if debug_mode and st.session_state['debug_logs']:
            st.subheader("Debug Logs")
            for log in st.session_state['debug_logs'][-10:]:
                st.text(log)

    uploaded_files = st.file_uploader(
        "Upload documents (txt/pdf)", type=["txt", "pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("Processing documents..."):
            for file in uploaded_files:
                try:
                    if file.type == "application/pdf":
                        pdf = PdfReader(file)
                        file_content = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                    else:
                        file_content = file.read().decode("utf-8")

                    chunks = split_text(file_content)
                    for i, chunk in enumerate(chunks):
                        emb = get_embedding(chunk)
                        if emb:
                            id_ = f"{file.name}_{i}_{abs(hash(chunk))}_{int(time.time())}"
                            meta = {"source": file.name, "chunk_index": i, "created_at": datetime.datetime.utcnow().isoformat()}
                            st.session_state["db_queue"].put_nowait({
                                "id": id_,
                                "document": chunk,
                                "embedding": emb,
                                "metadata": meta
                            })
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
            st.success("✅ Documents enqueued for DB storage")

    try:
        collection_count = collection.count()
        if collection_count > 0:
            st.info(f"📚 Knowledge base contains {collection_count} chunks")
    except Exception as e:
        st.warning(f"Error getting collection count: {e}")

    query = st.text_input("💬 Ask a question about your documents:")
    if query:
        with st.spinner("Processing..."):
            query_emb = get_embedding(query)
            if query_emb:
                results = collection.query(
                    query_embeddings=[query_emb],
                    n_results=min(3, collection.count()),
                    include=["documents", "metadatas"]
                )
                docs = results.get("documents", [[]])
                if docs and docs[0]:
                    context = " ".join(docs[0])
                    st.subheader("📖 Answer:")
                    answer_box = st.empty()
                    for partial in chat_with_ollama(context, query, model_choice):
                        answer_box.markdown(partial)
                else:
                    st.warning("No relevant documents found.")

# ---------------------- News Mode ----------------------
elif mode == "📰 News Search":
    topic = st.text_input("Enter a news topic:")
    if st.button("Fetch News") and NEWS_API_KEY:
        articles = fetch_news(topic, NEWS_API_KEY)
        if articles:
            st.success(f"✅ Retrieved {len(articles)} articles")
            for art in articles:
                for chunk in split_text(art):
                    emb = get_embedding(chunk)
                    if emb:
                        id_ = f"{topic}_{abs(hash(chunk))}_{int(time.time())}"
                        meta = {"source": "news", "topic": topic, "created_at": datetime.datetime.utcnow().isoformat()}
                        st.session_state["db_queue"].put_nowait({
                            "id": id_,
                            "document": chunk,
                            "embedding": emb,
                            "metadata": meta
                        })
        else:
            st.warning("⚠️ No articles found")

    query = st.text_input("💬 Ask a question about the news:")
    if query:
        query_emb = get_embedding(query)
        if query_emb:
            results = collection.query(
                query_embeddings=[query_emb],
                n_results=min(3, collection.count()),
                include=["documents", "metadatas"]
            )
            docs = results.get("documents", [[]])
            if docs and docs[0]:
                context = " ".join(docs[0])
                st.subheader("📰 Answer:")
                answer_box = st.empty()
                for partial in chat_with_ollama(context, query, model_choice):
                    answer_box.markdown(partial)
            else:
                st.warning("No relevant documents found.")
