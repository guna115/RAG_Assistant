# RAG_Assistant
# 📄 PDF Chat Assistant

A Streamlit-powered web app that allows you to **chat with your PDF file** using semantic search and retrieval-augmented generation (RAG).

This project uses:

✅ **Groq LLaMA-3.1-8b-instant** for LLM completions  
✅ **LangChain** for RAG orchestration  
✅ **Chroma vector store** for document retrieval  
✅ **Stable embeddings with FakeEmbeddings (no PyTorch crash)**  
✅ **Streamlit UI with long chat support**

---

## 🛠️ Features

✔ Upload a PDF and index it  
✔ Ask multiple questions one after another  
✔ Continuous chat history  
✔ Safe question answering (only from the uploaded PDF)

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/guna115/RAG_Assistant.git
cd RAG_Assistant
