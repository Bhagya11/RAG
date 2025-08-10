import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import google.generativeai as genai
import pandas as pd
from ragas import evaluate
from datasets import Dataset

# === Import all necessary RAGAS metrics ===
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision
)
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.llms import LangchainLLMWrapper

# === Configure Gemini API Key ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAJoievCdhnH4VUJjTVZ-Vkp1J3v1D53ao")
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY environment variable not set.")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)

# === Streamlit UI ===
st.set_page_config(page_title="📄 Chat with your PDF", layout="wide")
st.title("📄 Chat with your PDF using Gemini 2.5 Pro + RAGAS Evaluation")

# === Sidebar for RAGAS Metric Selection ===
st.sidebar.header("📊 RAGAS Evaluation Metrics")
selected_metrics_options = st.sidebar.multiselect(
    "Choose RAGAS Metrics to run:",
    options=[
        "Answer Relevancy",
        "Faithfulness",
        "Context Precision",
        "Context Recall"
    ],
    default=[
        "Answer Relevancy",
        "Faithfulness"
    ],
    help="Select which RAGAS metrics you want to calculate. More metrics may increase evaluation time."
)

# === Map metric names to RAGAS objects ===
metric_map = {
    "Answer Relevancy": answer_relevancy,
    "Faithfulness": faithfulness,
    "Context Precision": context_precision,
    "Context Recall": context_recall
}
selected_metrics = [metric_map[name] for name in selected_metrics_options]

uploaded_file = st.file_uploader("📤 Upload your PDF", type="pdf")
user_question = st.text_input("💬 Ask a question about the PDF")

# === Input for Ground Truth ===
ground_truth = st.text_area(
    "Ground Truth (Optional for Faithfulness/Relevancy, required for Recall/Precision)",
    help="Provide the correct answer to your question for a more complete RAGAS evaluation."
)

# === Gemini Embedding Wrapper ===
class GeminiEmbeddings(Embeddings):
    """Custom wrapper for Gemini Embeddings."""
    def embed_documents(self, texts):
        return [genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )["embedding"] for text in texts]

    def embed_query(self, text):
        return genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_query"
        )["embedding"]

# === Vector Store Builder ===
@st.cache_resource(show_spinner="📦 Building vector store...")
def build_vector_store(_chunks, _embedding_function):
    """Builds a FAISS vector store from document chunks."""
    return FAISS.from_documents(documents=_chunks, embedding=_embedding_function)

# === Main Logic ===
if uploaded_file and user_question:
    if not selected_metrics:
        st.warning("Please select at least one RAGAS metric from the sidebar to perform evaluation.")
        st.stop()
    
    # Filter metrics that require ground truth if no ground truth is provided
    final_selected_metrics = selected_metrics.copy()
    if not ground_truth:
        metrics_to_remove = [context_precision, context_recall]
        final_selected_metrics = [m for m in final_selected_metrics if m not in metrics_to_remove]
        if any(m in [context_precision, context_recall] for m in selected_metrics):
            st.warning("Ground Truth is missing. 'Context Precision' and 'Context Recall' metrics will be skipped.")

    if not final_selected_metrics:
        st.info("No metrics selected or available for evaluation with the provided inputs.")
        st.stop()

    with st.spinner("⚙️ Processing PDF and finding answer..."):
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            # Load and split PDF
            loader = PyPDFLoader(tmp_file_path)
            pages = loader.load_and_split()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(pages)

            # Create embeddings + vector store
            gemini_embeddings = GeminiEmbeddings()
            vectorstore = build_vector_store(chunks, gemini_embeddings)

            # Search top matching chunks
            docs = vectorstore.similarity_search(user_question, k=4)
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = f"""You are a helpful assistant answering based only on the document context.

Context from the PDF:
---
{context}
---

Question: {user_question}

Answer:"""

            model = genai.GenerativeModel("gemini-2.5-pro")
            response = model.generate_content(prompt)

            # === Display Answer ===
            st.markdown("### 📘 Answer")
            st.write(response.text)

            # === Display Context Chunks ===
            with st.expander("📄 Show retrieved context chunks"):
                for i, doc in enumerate(docs):
                    st.markdown(f"**Chunk {i+1} (Page {doc.metadata.get('page', 'N/A')})**")
                    st.info(doc.page_content)

            # === RAGAS Evaluation ===
            st.markdown("### 📊 RAGAS Evaluation")
            ragas_data = {
                "question": [user_question],
                "answer": [response.text],
                "contexts": [[doc.page_content for doc in docs]],
                "ground_truths": [ground_truth if ground_truth else ""]
            }
            dataset = Dataset.from_dict(ragas_data)

            with st.spinner("📈 Running RAGAS evaluation..."):
                # Define LLM for RAGAS evaluation
                ragas_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GEMINI_API_KEY)
                ragas_llm_wrapper = LangchainLLMWrapper(ragas_llm)

                try:
                    results = evaluate(
                        dataset,
                        metrics=final_selected_metrics,
                        llm=ragas_llm_wrapper,
                        embeddings=gemini_embeddings
                    )

                    results_df = results.to_pandas()
                    st.write("Full RAGAS results dataframe:")
                    st.dataframe(results_df)

                    st.write("---")
                    st.write("### Average Scores")
                    average_scores = results_df.iloc[0].items() if not results_df.empty else []
                    for metric_name, score in average_scores:
                        try:
                            formatted_score = f"{score:.3f}"
                            st.write(f"**{metric_name.replace('_', ' ').title()}**: {formatted_score}")
                        except (TypeError, ValueError):
                            st.write(f"**{metric_name.replace('_', ' ').title()}**: Could not calculate score.")

                except Exception as eval_e:
                    st.error(f"❌ An error occurred during RAGAS evaluation: {eval_e}")
                    st.warning("This can sometimes happen due to a parsing issue with the LLM's output or missing ground truth. The rest of the app should still function.")

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
        finally:
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
