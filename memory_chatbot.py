# =========================================================
# IMPORTS
# =========================================================
# Core Python libraries
import os
import json
from datetime import datetime
from typing import List, Dict, Any

# Third-party libraries
import faiss                     # Vector database for similarity search
import numpy as np               # Numerical operations
import streamlit as st           # UI framework
import requests                  # API calls
from dotenv import load_dotenv   # Environment variable management


# =========================================================
# ENVIRONMENT VARIABLES & PATH SETUP
# =========================================================
# Load variables from .env file
load_dotenv()

# Directory where memory-related files will be stored
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Paths for FAISS index, metadata, and conversation history
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")
FAISS_META_PATH = os.path.join(DATA_DIR, "faiss_index_metadata.json")
HISTORY_PATH = os.path.join(DATA_DIR, "conversation_history.json")

# Load API key securely
API_KEY = os.getenv("EURIAI_API_KEY")
if not API_KEY:
    st.error("EURIAI_API_KEY not found in .env file")
    st.stop()


# =========================================================
# APPLICATION SETTINGS
# =========================================================
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-nano"
VECTOR_DIM = 1536                # Dimension of embedding vectors
TOP_K = 5                        # Number of relevant memories to retrieve
TEMPERATURE = 0.7
MAX_TOKENS = 800
HISTORY_LIMIT = 50               # Short-term memory limit


# =========================================================
# EMBEDDINGS CLASS (EURI API)
# =========================================================
class EuriaiEmbeddings:
    """
    Converts text into vector embeddings using EURI Embeddings API.
    These embeddings are stored in FAISS for similarity search.
    """
    URL = "https://api.euron.one/api/v1/euri/embeddings"

    def embed(self, text: str) -> List[float]:
        response = requests.post(
            self.URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "input": text,
                "model": EMBEDDING_MODEL
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]


# =========================================================
# CHAT COMPLETION CLASS (EURI API)
# =========================================================
class EuriaiChat:
    """
    Sends prompt (user input + retrieved memories) to the LLM
    and returns the generated response.
    """
    URL = "https://api.euron.one/api/v1/euri/chat/completions"

    def generate(self, prompt: str) -> str:
        response = requests.post(
            self.URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": CHAT_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


# =========================================================
# VECTOR STORE (LONG-TERM MEMORY USING FAISS)
# =========================================================
class MemoryVectorStore:
    """
    Stores long-term memory as embeddings using FAISS.
    Supports adding new memories and retrieving relevant ones.
    """

    def __init__(self):
        self.embedder = EuriaiEmbeddings()

        # Load existing FAISS index if available
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_META_PATH):
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.index = faiss.IndexFlatL2(VECTOR_DIM)
            self.metadata = []

    def save(self):
        """Persist FAISS index and metadata to disk"""
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        with open(FAISS_META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def add(self, text: str, meta: Dict[str, Any]):
        """
        Add a new memory:
        - Convert text to embedding
        - Store in FAISS
        - Save metadata with timestamp
        """
        vector = np.array([self.embedder.embed(text)], dtype=np.float32)
        self.index.add(vector)

        self.metadata.append({
            "text": text,
            "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
            **meta
        })
        self.save()

    def search(self, query: str, k: int = TOP_K):
        """
        Retrieve top-k relevant memories based on vector similarity.
        """
        if not self.metadata:
            return []

        vector = np.array([self.embedder.embed(query)], dtype=np.float32)
        _, indices = self.index.search(vector, min(k, len(self.metadata)))

        return [self.metadata[i] for i in indices[0]]


# =========================================================
# CONVERSATION MEMORY (SHORT-TERM)
# =========================================================
class ConversationMemory:
    """
    Stores recent conversation messages with timestamps.
    This memory can be cleared by the user.
    """

    def __init__(self):
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                self.history = json.load(f)
        else:
            self.history = []

    def save(self):
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def add(self, role: str, content: str):
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p")
        })
        self.history = self.history[-HISTORY_LIMIT:]
        self.save()

    def clear(self):
        self.history = []
        self.save()


# =========================================================
# MAIN CHATBOT LOGIC (RAG PIPELINE)
# =========================================================
class MemoryChatbot:
    """
    Combines:
    - Short-term memory
    - Long-term FAISS memory
    - Prompt generation (RAG)
    """

    def __init__(self):
        self.chat = EuriaiChat()
        self.vector_store = MemoryVectorStore()
        self.conversation = ConversationMemory()
        self.identity = ""

    def set_identity(self, identity: str):
        """Store user identity permanently in vector memory"""
        self.identity = identity
        self.vector_store.add(
            f"User identity: {identity}",
            {"type": "identity"}
        )

    def respond(self, user_input: str):
        """
        Full RAG flow:
        1. Store user input
        2. Retrieve relevant memories
        3. Generate prompt
        4. Get response from LLM
        5. Store interaction back into memory
        """
        self.conversation.add("user", user_input)

        memories = self.vector_store.search(user_input)
        memory_block = "\n".join(m["text"] for m in memories)

        prompt = f"""
You are a memory-based assistant.

User identity:
{self.identity}

Relevant memories:
{memory_block}

User question:
{user_input}
"""

        response = self.chat.generate(prompt)

        self.conversation.add("assistant", response)

        self.vector_store.add(
            f"User: {user_input}\nAssistant: {response}",
            {"type": "interaction"}
        )

        suggestions = [
            {
                "text": m["text"],
                "timestamp": m.get("timestamp", ""),
                "type": m.get("type", "memory")
            }
            for m in memories
        ]

        return response, suggestions


# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config("🧠 Memory Chatbot", "🧠", layout="wide")

if "bot" not in st.session_state:
    st.session_state.bot = MemoryChatbot()

if "suggestions" not in st.session_state:
    st.session_state.suggestions = []

st.title("🧠 Memory Chatbot")


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.subheader("User Identity")
    identity_input = st.text_area(
        "Set / Update Identity",
        value=st.session_state.bot.identity,
        height=100
    )

    if st.button("Save Identity"):
        st.session_state.bot.set_identity(identity_input)
        st.success("Identity saved permanently")

    st.markdown("---")

    st.subheader("Custom Memory")
    custom_memory = st.text_area("Add memory", height=120)

    if st.button("Store Memory"):
        st.session_state.bot.vector_store.add(
            custom_memory,
            {"type": "custom"}
        )
        st.success("Memory stored")

    st.markdown("---")

    if st.button("Clear Conversation"):
        st.session_state.bot.conversation.clear()
        st.session_state.suggestions = []
        st.rerun()


# ---------------- CHAT HISTORY ----------------
for msg in st.session_state.bot.conversation.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        st.caption(msg.get("timestamp", ""))


# ---------------- USER INPUT ----------------
user_input = st.chat_input("Type your message...")
if user_input:
    reply, suggestions = st.session_state.bot.respond(user_input)
    st.session_state.suggestions = suggestions
    st.rerun()


# ---------------- SUGGESTIONS PANEL ----------------
if st.session_state.suggestions:
    st.markdown("---")
    st.subheader("🔎 Suggestions & Related Memories")
    for i, s in enumerate(st.session_state.suggestions, 1):
        with st.expander(f"Memory {i} ({s.get('type', 'memory')})"):
            st.write(s["text"])
            if s.get("timestamp"):
                st.caption(f"Stored on: {s['timestamp']}")
