# Memory-Based-ChatBot
# 🧠 Memory-Based Chatbot with Long-Term Context (FAISS + Streamlit)

This project implements a **memory-augmented conversational chatbot** that can remember **user identity, past conversations, and private contextual information across sessions**.

Unlike a general chatbot that sends only the current user input to an LLM, this system **retrieves the top-5 most relevant historical memories from a FAISS vector database** and injects them into the prompt, enabling **context-aware and personalized responses**.



## 🚀 Key Features

✔ Long-term memory using **FAISS vector database**  
✔ Short-term conversation memory with **timestamps**  
✔ Persistent **user identity memory**  
✔ **Top-5 relevant memory retrieval** per query  
✔ **RAG-based prompt generation**  
✔ Suggestions & related memories display  
✔ Custom memory storage  
✔ Streamlit chat-style UI  
✔ Secure API key handling using `.env`



## 🤖 General Chatbot vs Memory-Based Chatbot

| Feature | General Chatbot | Memory-Based Chatbot (This Project) |
|------|----------------|--------------------------------------|
| Input to LLM | Only current message | Current message + retrieved memories |
| Memory | Session-based | Long-term + short-term |
| Context after refresh | ❌ Lost | ✅ Preserved |
| Personalization | ❌ No | ✅ Yes |
| Vector DB | ❌ Not used | ✅ FAISS |
| RAG | ❌ No | ✅ Yes |
| Suggestions | ❌ No | ✅ Yes |
| Timestamps | ❌ No | ✅ Yes |



## 🧠 How the Chatbot Works (Architecture Flow)

### 1️⃣ User Input
The user enters a message using the Streamlit chat interface.



### 2️⃣ Embedding Generation
The user input is converted into a numerical vector using the **EURI Embeddings API**.



### 3️⃣ Memory Retrieval (FAISS)
- The embedding is searched against stored vectors in **FAISS**
- The **top 5 most relevant memories** are retrieved
- These memories may include:
  - Previous conversations
  - User identity
  - Custom user-defined memories
  - Past private context



### 4️⃣ Prompt Generation (RAG)
A **Retrieval-Augmented Generation (RAG)** prompt is constructed using:
- User identity
- Retrieved relevant memories
- Current user input

This allows the LLM to generate **context-aware responses**.



### 5️⃣ Response Generation
The prompt is sent to the **EURI Chat Completion API**, which generates the assistant response.



### 6️⃣ Memory Update
- The interaction is stored back into FAISS
- Conversation history is saved with **date & time**
- Memory persists across sessions


## 🔍 Types of Memory Used

### 🔹 Short-Term Memory
- Current conversation
- Stored with timestamps
- Can be cleared by the user

### 🔹 Long-Term Memory
- Stored in FAISS
- Includes identity, interactions, custom memory
- **Not removed when conversation is cleared**
- Enables long-term personalization

---

## ⏱️ Timestamped Conversations

Each user and assistant message is stored with:
- Date
- Time

This improves traceability and conversation analysis.

---

## 💡 Suggestions & Related Memories

For every query:
- The top-5 similar memories are retrieved
- Displayed as **Suggestions & Related Memories**
- Helps explain how the response was generated

---

## 🏗️ Tech Stack

- **Python 3.10+**
- **Streamlit** – Frontend UI
- **FAISS (CPU)** – Vector similarity search
- **NumPy** – Vector operations
- **Requests** – API communication
- **python-dotenv** – Environment variables
- **EURI AI APIs** – Embeddings & Chat Completion



