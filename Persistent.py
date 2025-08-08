

import streamlit as st
import tempfile
import os
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
import google.generativeai as genai

# --- Configuration ---
# It's recommended to set the API key as an environment variable
try:
    # Attempt to get the API key from environment variables or Streamlit secrets
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY","AIzaSyAJoievCdhnH4VUJjTVZ-Vkp1J3v1D53ao")
    if not GEMINI_API_KEY:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except (KeyError, TypeError):
    st.error("🔴 GEMINI_API_KEY environment variable or Streamlit secret not set.")
    st.stop()

# Define the path for the persistent Chroma database
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "pdf_documents"

st.set_page_config(page_title="📄 Persistent Chat with PDFs", layout="wide")


# --- Embedding Function ---
class GeminiEmbeddings(Embeddings):
    """A wrapper for the Gemini embedding model."""
    def embed_documents(self, texts):
        return [genai.embed_content(model="models/embedding-001", content=text, task_type="retrieval_document")["embedding"] for text in texts]

    def embed_query(self, text):
        return genai.embed_content(model="models/embedding-001", content=text, task_type="retrieval_query")["embedding"]


# --- Helper Functions ---
def format_docs(docs):
    """Helper function to format retrieved documents into a string."""
    return "\n\n".join(doc.page_content for doc in docs)

# --- Main Application Logic ---
def main():
    st.title("📄 Chat With Your PDFs (with Persistent Memory)")
    st.markdown("""
    Welcome! Your uploaded documents are now saved and will be available even after you refresh the page.
    """)

    # Initialize ChromaDB client and collection
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    embeddings = GeminiEmbeddings()
    vector_store = Chroma(
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    # --- Sidebar for Document Management ---
    with st.sidebar:
        st.header("📁 Document Management")

        # Get the list of already processed files from ChromaDB's metadata
        # This is our way of "loading" the state on startup
        try:
            collection_metadata = vector_store.get(include=["metadatas"])
            processed_files = sorted(list(set(meta['source'] for meta in collection_metadata['metadatas'])))
        except (IndexError, ValueError): # Handles case where DB is empty
            processed_files = []

        # Initialize session state variables if they don't exist
        if "selected_docs" not in st.session_state:
            st.session_state.selected_docs = processed_files.copy()
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        uploaded_files = st.file_uploader(
            "📤 Upload new PDFs here",
            type="pdf",
            accept_multiple_files=True
        )

        if uploaded_files:
            new_files_to_process = [file for file in uploaded_files if file.name not in processed_files]
            if new_files_to_process:
                with st.spinner("Processing new PDFs... This might take a moment."):
                    for file in new_files_to_process:
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                                tmp_file.write(file.getvalue())
                                loader = PyPDFLoader(tmp_file.name)
                                pages = loader.load_and_split()
                                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                                chunks = text_splitter.split_documents(pages)

                                # Add source metadata to each chunk
                                for chunk in chunks:
                                    chunk.metadata["source"] = file.name

                                vector_store.add_documents(chunks)
                                st.sidebar.success(f"✅ '{file.name}' processed and saved!")
                                if file.name not in st.session_state.selected_docs:
                                     st.session_state.selected_docs.append(file.name)
                            os.remove(tmp_file.name)
                        except Exception as e:
                            st.sidebar.error(f"Error with {file.name}: {e}")
                st.rerun() # Rerun to update the list of processed files

        if processed_files:
            st.markdown("---")
            st.subheader("🔎 Select & Manage Documents")

            for file_name in processed_files:
                col1, col2 = st.columns([0.85, 0.15])
                with col1:
                    is_selected = st.checkbox(
                        file_name,
                        value=(file_name in st.session_state.selected_docs),
                        key=f"select_{file_name}"
                    )
                    if is_selected and file_name not in st.session_state.selected_docs:
                        st.session_state.selected_docs.append(file_name)
                    elif not is_selected and file_name in st.session_state.selected_docs:
                        st.session_state.selected_docs.remove(file_name)
                with col2:
                    if st.button("🗑️", key=f"delete_{file_name}", help=f"Delete '{file_name}' permanently"):
                        # Delete from ChromaDB
                        ids_to_delete = [id for id, meta in zip(collection_metadata['ids'], collection_metadata['metadatas']) if meta['source'] == file_name]
                        if ids_to_delete:
                            vector_store.delete(ids=ids_to_delete)
                        # Update session state
                        if file_name in st.session_state.selected_docs:
                            st.session_state.selected_docs.remove(file_name)
                        st.rerun()
        else:
            st.info("Upload PDF files to begin your persistent library.")

    # --- Main Chat Interface ---
    for author, content in st.session_state.chat_history:
        with st.chat_message(author):
            st.markdown(content)

    if user_question := st.chat_input("Ask a question about your selected documents..."):
        st.session_state.chat_history.append(("user", user_question))
        with st.chat_message("user"):
            st.markdown(user_question)

        if not st.session_state.selected_docs:
            st.session_state.chat_history.append(("assistant", "Please select at least one document to search."))
            with st.chat_message("assistant"):
                st.warning("Please select at least one document to search.")
            st.rerun()

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Create a retriever that filters by selected source documents
                    retriever = vector_store.as_retriever(
                        search_kwargs={
                            "k": 10,
                            'filter': {'source': {'$in': st.session_state.selected_docs}}
                        }
                    )
                    retrieved_docs = retriever.invoke(user_question)

                    if not retrieved_docs:
                        st.warning("Could not find relevant information in the selected documents for your query.")
                        st.session_state.chat_history.append(("assistant", "I couldn't find any relevant information in the selected documents to answer your question."))
                        st.rerun()

                    # Create the RAG chain
                    template = """You are a helpful assistant specialized in answering questions based on the provided document context.
                    Your goal is to be precise and base your answers strictly on the information given.
                    If the context doesn't contain the answer, state that clearly.

                    Context from the PDF(s):
                    ---
                    {context}
                    ---

                    Question: {question}

                    Answer:"""
                    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
                    model = genai.GenerativeModel("gemini-2.5-pro")

                    context_str = format_docs(retrieved_docs)
                    full_prompt = prompt.format(context=context_str, question=user_question)
                    response = model.generate_content(full_prompt)
                    ai_response = response.text

                    st.markdown(ai_response)
                    st.session_state.chat_history.append(("assistant", ai_response))

                    with st.expander("📄 Show retrieved context chunks"):
                         for doc in retrieved_docs:
                            st.info(f"**Chunk from '{doc.metadata.get('source', 'N/A')}' (Page {doc.metadata.get('page', 'N/A')})**")
                            st.write(doc.page_content)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
