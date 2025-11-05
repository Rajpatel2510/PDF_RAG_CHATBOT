import streamlit as st
import tempfile
from pdf_handler import extract_text
from chunk_text import text_chunker
from embedder import Embedder
from llm import GeminiLLM
from chromadb_handler import ChromaDBHandler
from self_rag import SelfRAG

st.title("Chatbot")

vector = ChromaDBHandler()
embedder = Embedder()

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

uploaded_file = st.file_uploader("choose file", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name
        pdf_name = uploaded_file.name.replace(" ", "_").replace(".", "_")
        st.session_state.pdf_name = pdf_name

    try:
        extracted_text = extract_text(file_path)
        chunks = text_chunker(extracted_text)
        texts = [c.page_content if hasattr(c,"page_content") else c for c in chunks]
        embeddings = embedder.encode(texts)

        # st.success(f"Chunks: {len(chunks)}, Embeddings: {len(embeddings)}")

        vector.add_data(pdf_name, texts, embeddings)
        st.success("Data inserted into ChromaDB")

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.warning("Please upload a PDF file.")

user_question = st.text_input("Ask a question about the resume:")

if st.button("Search"):
    if not st.session_state.pdf_name:
        st.warning("Upload a file first.")
    else:
        chunks = vector.search_similar_chunks(st.session_state.pdf_name, user_question, 5)
        for i, (text, score) in enumerate(chunks):
            st.write(f"Match {i}: {text} (score: {score})")


llm = GeminiLLM()
selfrag = SelfRAG(llm)
if st.button("Generate Answer"):

    question = user_question.strip()
    if not question:
        st.warning("Enter a question.")
        st.stop()

    context_pairs = vector.search_similar_chunks(st.session_state.pdf_name, question)

    if not context_pairs:
        st.error("No relevant chunks found.")
        st.stop()

    chunks = [c[0] for c in context_pairs]

    # ✅ Step-1: Evaluate relevance (Self-RAG)
    evaluation = selfrag.evaluate_context(question, chunks)
    # st.write("Self-RAG Score:", evaluation)
    st.markdown(f"**Self-RAG Evaluation:** {evaluation}")


    if "bad" in evaluation.lower():
        st.warning("Low context confidence, refining retrieval…")
        context_pairs = vector.search_similar_chunks(st.session_state.pdf_name, question)
        chunks = [c[0] for c in context_pairs]

    # ✅ Step-2: Final answer grounded to context
    answer = selfrag.final_answer(question, chunks)
    st.markdown("### Final Answer")
    st.write(answer)
