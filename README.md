RAG

# 📄 Chat with Your PDF using Gemini 2.5 Pro

This is a **minimal, fast, and efficient** Streamlit app that lets you upload a PDF and ask questions about its content. It uses **Google Gemini 2.5 Pro** for generating answers and **FAISS** for vector similarity search.

---

## 🚀 Features

✅ Upload a **single PDF**  
✅ Ask **questions** based on the PDF's content  
✅ Uses **Google Gemini** for embeddings and answers  
✅ Fast in-memory **FAISS** vector store  
✅ See **retrieved context chunks** used for answering  
✅ Clean and simple **Streamlit UI**

---

## 🧠 How It Works

1. **Upload PDF** ➜ It gets split into chunks using LangChain.  
2. **Embed chunks** ➜ Uses `models/embedding-001` from Gemini.  
3. **Build FAISS store** ➜ Documents stored in memory.  
4. **Ask question** ➜ Retrieves top relevant chunks.  
5. **Prompt Gemini 2.5 Pro** ➜ Answers based on those chunks.

---
<img width="1350" height="675" alt="2" src="https://github.com/user-attachments/assets/b9a5a838-a838-4b2f-96d1-62f7b1d13b9a" />
<img width="1273" height="624" alt="1" src="https://github.com/user-attachments/assets/11218907-6811-4edc-aa92-6888cf923a5d" />


RAG_DB

# 🤖 Chat with Multiple PDFs using Gemini 2.5 Pro

A powerful and intuitive Streamlit application that allows you to **upload multiple PDF files**, chat with them, and get intelligent answers using **Google Gemini 2.5 Pro**, **LangChain**, and **FAISS** for semantic search.

---

## 🚀 Features

- 📄 **Upload multiple PDFs** and manage them easily
- 🧠 **Ask questions** across selected documents
- 🧷 Uses **Gemini 2.5 Pro** for embedding + generation
- 📚 Built-in **vector database (FAISS)** per document
- 💬 **Chat memory** to maintain conversation context
- 🔍 View context chunks used in each answer
- ✅ Simple and responsive **Streamlit UI**

---

## 🧠 How It Works

1. Upload PDFs ➜ Each file is split and embedded.
2. Store in FAISS ➜ Vector store is built per document.
3. Ask a question ➜ Top-k relevant chunks are retrieved.
4. RAG pipeline ➜ Gemini 2.5 Pro answers using only the context.

---



<img width="1348" height="628" alt="3" src="https://github.com/user-attachments/assets/5252cab5-1eb3-44a8-b968-178578ac9899" />


RAG_persistent Memory

# 🧠 Multi-PDF Chatbot using Gemini 2.5 + ChromaDB (RAG Pipeline)

This is a RAG-based (Retrieval-Augmented Generation) chatbot built with **Streamlit**, powered by **Gemini Pro 2.5**, and backed by a **persistent ChromaDB vector store**. You can upload and chat with multiple PDFs using Google's latest Gemini LLM and retrieve answers grounded in your documents.

---

## 🚀 Features

- ✅ Chat with multiple PDF files
- ✅ Uses Google's **Gemini Pro 2.5** model
- ✅ Vector storage with **ChromaDB** (persistent local memory)
- ✅ Intelligent document retrieval using embeddings
- ✅ Metadata filtering for document-specific answers
- ✅ Streamlit UI with session memory
- ✅ Supports context-aware question answering

---




<img width="1365" height="663" alt="4" src="https://github.com/user-attachments/assets/89a4ea29-2c62-4df9-a3c7-2544d35d3853" />

