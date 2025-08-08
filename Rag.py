import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import google.generativeai as genai

# === Configure Gemini API Key ===
# It's recommended to set the API key as an environment variable
# for security reasons.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY","AIzaSyAJoievCdhnH4VUJjTVZ-Vkp1J3v1D53ao")
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY environment variable not set.")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)


# === Streamlit UI ===
st.set_page_config(page_title="📄 Chat with your PDF", layout="wide")
st.title("📄 Chat with your PDF using Gemini 2.5 Pro")

uploaded_file = st.file_uploader("📤 Upload your PDF", type="pdf")
user_question = st.text_input("💬 Ask a question about the PDF")


# === Gemini Embedding Function ===
# We create a class that inherits from LangChain's Embeddings
# to integrate smoothly with LangChain's vector stores.
class GeminiEmbeddings(Embeddings):
    """
    A wrapper for the Gemini embedding model that conforms to
    the LangChain Embeddings interface.
    """
    def embed_documents(self, texts):
        """Embeds a list of documents."""
        return [genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )["embedding"] for text in texts]

    def embed_query(self, text):
        """Embeds a single query."""
        return genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_query"
        )["embedding"]


# === Vector Store Builder (SIMPLIFIED) ===
@st.cache_resource(show_spinner="Building vector store...")
def build_vector_store(_chunks, _embedding_function):
    """
    Builds and caches a FAISS vector store from document chunks.
    The underscore prefix on arguments tells Streamlit to not hash them,
    as they can be large or complex objects.
    """
    vectorstore = FAISS.from_documents(
        documents=_chunks,
        embedding=_embedding_function
    )
    return vectorstore

# === Main Execution ===
if uploaded_file and user_question:
    with st.spinner("⚙️ Processing PDF and finding answer..."):
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            # Load and split PDF
            loader = PyPDFLoader(tmp_file_path)
            pages = loader.load_and_split() # More efficient than load() then split()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(pages)

            # Create an instance of our embedding class
            gemini_embeddings = GeminiEmbeddings()

            # Build vector store using the simplified and cached function
            vectorstore = build_vector_store(chunks, gemini_embeddings)

            # Search top matching chunks
            docs = vectorstore.similarity_search(user_question, k=4)

            # Prepare context and prompt
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = f"""You are a helpful assistant specialized in answering questions based on the provided document context. Your goal is to be precise and base your answers strictly on the information given.

Context from the PDF:
---
{context}
---

Question: {user_question}

Answer:"""

            model = genai.GenerativeModel("gemini-2.5-pro")
            response = model.generate_content(prompt)

            # Display result
            st.markdown("### 📘 Answer")
            st.write(response.text)

            # Optional: show context
            with st.expander("📄 Show retrieved context chunks"):
                for i, doc in enumerate(docs):
                    st.markdown(f"**Chunk {i+1} (Source: Page {doc.metadata.get('page', 'N/A')})**")
                    st.info(doc.page_content)

        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            # Clean up the temporary file
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)