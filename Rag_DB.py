import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import google.generativeai as genai

# === Configure Gemini API Key ===
# It's recommended to set the API key as an environment variable
# for security reasons.
# You can do this in your terminal: export GEMINI_API_KEY="YOUR_API_KEY"
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY","AIzaSyAJoievCdhnH4VUJjTVZ-Vkp1J3v1D53ao")
    if not GEMINI_API_KEY:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except (KeyError, TypeError):
    st.error("GEMINI_API_KEY environment variable or Streamlit secret not set.")
    st.stop()


# === Streamlit UI Configuration ===
st.set_page_config(page_title="📄 Chat with Multiple PDFs", layout="wide")


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

# === Helper Functions ===
@st.cache_resource(show_spinner="Processing PDF...")
def process_pdf(file):
    """
    Loads, splits, and creates a vector store for a single uploaded PDF file.
    The @st.cache_resource decorator ensures this function is only run once
    per file, and the result is cached.
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name

        # Load and split PDF
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(pages)

        # Create embeddings and vector store
        embeddings = GeminiEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)

        return vectorstore

    except Exception as e:
        st.error(f"Error processing {file.name}: {e}")
        return None
    finally:
        # Clean up the temporary file
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

def format_docs(docs):
    """Helper function to format retrieved documents into a string."""
    return "\n\n".join(doc.page_content for doc in docs)

# === Main Application Logic ===
def main():
    st.title("📄 Chat With Your PDFs using Gemini 2.5 Pro")
    st.markdown("""
    Welcome! Upload one or more PDFs, select which ones to use, and ask questions about their content.
    """)

    # --- Sidebar for Document Management ---
    with st.sidebar:
        st.header("📁 Document Management")

        # Initialize session state variables
        if "vector_stores" not in st.session_state:
            st.session_state.vector_stores = {}
        if "selected_docs" not in st.session_state:
            st.session_state.selected_docs = []
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []


        uploaded_files = st.file_uploader(
            "📤 Upload your PDFs here",
            type="pdf",
            accept_multiple_files=True
        )

        if uploaded_files:
            for file in uploaded_files:
                if file.name not in st.session_state.vector_stores:
                    vector_store = process_pdf(file)
                    if vector_store:
                        st.session_state.vector_stores[file.name] = vector_store
                        # Automatically select newly uploaded files
                        st.session_state.selected_docs.append(file.name)
                        st.sidebar.success(f"✅ '{file.name}' processed!")

        if st.session_state.vector_stores:
            st.markdown("---")
            st.subheader("🔎 Select & Manage Documents")

            # Use a copy of the list to allow modification during iteration
            for file_name in list(st.session_state.vector_stores.keys()):
                col1, col2 = st.columns([0.85, 0.15]) # Adjust column ratio

                with col1:
                    # Checkbox for selection
                    is_selected = st.checkbox(
                        file_name,
                        value=(file_name in st.session_state.selected_docs),
                        key=f"select_{file_name}" # Unique key for checkbox
                    )
                    if is_selected and file_name not in st.session_state.selected_docs:
                        st.session_state.selected_docs.append(file_name)
                    elif not is_selected and file_name in st.session_state.selected_docs:
                        st.session_state.selected_docs.remove(file_name)

                with col2:
                    # **FIX:** Use st.button with a unique key for deletion
                    if st.button("🗑️", key=f"delete_{file_name}", help=f"Delete '{file_name}'"):
                        del st.session_state.vector_stores[file_name]
                        if file_name in st.session_state.selected_docs:
                            st.session_state.selected_docs.remove(file_name)
                        st.rerun() # Rerun to update the UI immediately
        else:
            st.info("Upload PDF files to begin.")


    # --- Main Chat Interface ---
    # Display chat messages from history
    for author, content in st.session_state.chat_history:
        with st.chat_message(author):
            st.markdown(content)

    if user_question := st.chat_input("Ask a question about your selected documents..."):
        # Add user message to chat history
        st.session_state.chat_history.append(("user", user_question))
        with st.chat_message("user"):
            st.markdown(user_question)

        # Check if any documents are selected
        if not st.session_state.selected_docs:
            with st.chat_message("assistant"):
                st.warning("Please upload and select at least one document before asking a question.")
            st.session_state.chat_history.append(("assistant", "Please upload and select at least one document."))
            st.rerun()

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Retrieve relevant documents from selected vector stores
                    all_retrieved_docs = []
                    for doc_name in st.session_state.selected_docs:
                        retriever = st.session_state.vector_stores[doc_name].as_retriever(search_kwargs={"k": 5})
                        all_retrieved_docs.extend(retriever.invoke(user_question))

                    if not all_retrieved_docs:
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

                    # Simple RAG chain implementation
                    context_str = format_docs(all_retrieved_docs)
                    full_prompt = prompt.format(context=context_str, question=user_question)

                    response = model.generate_content(full_prompt)
                    ai_response = response.text

                    st.markdown(ai_response)
                    st.session_state.chat_history.append(("assistant", ai_response))

                    # Optional: show context
                    with st.expander("📄 Show retrieved context chunks"):
                         for i, doc in enumerate(all_retrieved_docs):
                            # Attempt to get a clean file name
                            source_name = "N/A"
                            if 'source' in doc.metadata:
                                source_name = os.path.basename(doc.metadata['source'])

                            st.info(f"**Chunk from '{source_name}' (Page {doc.metadata.get('page', 'N/A')})**")
                            st.write(doc.page_content)

                except Exception as e:
                    st.error(f"An error occurred while generating the answer: {e}")


if __name__ == "__main__":
    main()

