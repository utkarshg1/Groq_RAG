import streamlit as st
import os
import shutil
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# -----------------
# Load API keys safely from Streamlit secrets
# -----------------
try:
    cohere_api_key = st.secrets["COHERE_API_KEY"]
    groq_api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error(
        "Missing API key(s) in Streamlit secrets! Please add COHERE_API_KEY and GROQ_API_KEY."
    )
    st.stop()

os.environ["COHERE_API_KEY"] = cohere_api_key
os.environ["GROQ_API_KEY"] = groq_api_key

# -----------------
# Initialize LLM + Embeddings
# -----------------
embedding_model = CohereEmbeddings(model="embed-english-v3.0")  # type: ignore
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5, max_tokens=500)


# -----------------
# Helpers
# -----------------
def load_pdf_and_split(path: str):
    loader = PyPDFLoader(path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(docs)


@st.cache_resource(show_spinner="Creating vectorstore...")
def get_vectorstore(splits, persist_dir):
    """Cache vectorstore to avoid re-computation when same file is uploaded"""
    return Chroma.from_documents(
        splits, embedding=embedding_model, persist_directory=persist_dir
    )


def get_response(retriever, query: str):
    """RAG pipeline using stuff chain with long-answer friendly prompt"""
    system_prompt = (
        "You are a helpful assistant. Use the retrieved context to answer the question "
        "as completely as possible. If the answer is not in the context, say that you don't know, "
        "but always explain your reasoning based on the available information. "
        "Provide detailed answers and examples when relevant.\n\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": query})
    return response["answer"]


def list_documents():
    """Return list of all persisted PDFs"""
    return [
        folder.replace("chroma_db_", "")
        for folder in os.listdir(".")
        if folder.startswith("chroma_db_")
    ]


# -----------------
# Streamlit App
# -----------------
st.set_page_config(page_title="RAG-Utkarsh", layout="wide")
st.title("RAG Chatbot - Utkarsh")

# Initialize session state
if "db" not in st.session_state:
    st.session_state.db = None
if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = None

query = st.text_input("Ask a question about the document")
submit = st.button("Answer")

# -----------------
# Sidebar: Upload & Document Selection
# -----------------
with st.sidebar:
    st.header("Documents & Upload")

    # Upload new PDF
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file:
        temp_file = f"./{uploaded_file.name}"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getvalue())
        if st.button("Create Embeddings"):
            with st.spinner("Processing..."):
                splits = load_pdf_and_split(temp_file)
                persist_dir = f"./chroma_db_{uploaded_file.name}"
                st.session_state.db = get_vectorstore(splits, persist_dir=persist_dir)
                st.session_state.selected_doc = uploaded_file.name
                st.success(f"Embeddings for '{uploaded_file.name}' created & stored!")

    # List persisted PDFs
    docs = list_documents()
    if docs:
        selected_doc = st.selectbox(
            "Select Document",
            docs,
            index=(
                docs.index(st.session_state.selected_doc)
                if st.session_state.selected_doc in docs
                else 0
            ),
        )
        st.session_state.selected_doc = selected_doc
        st.session_state.db = Chroma(
            persist_directory=f"./chroma_db_{selected_doc}",
            embedding_function=embedding_model,
        )
    else:
        st.info("No persisted documents found. Upload a PDF to get started.")

    # Clear all embeddings
    if st.button("Clear All Embeddings"):
        for folder in os.listdir("."):
            if folder.startswith("chroma_db_"):
                shutil.rmtree(folder)
        st.session_state.db = None
        st.session_state.selected_doc = None
        st.success("All embeddings cleared!")

# -----------------
# Handle QA
# -----------------
if submit:
    if st.session_state.db is None:
        st.error(
            "Please select a document or upload a PDF and create embeddings first."
        )
    else:
        with st.spinner("Responding..."):
            retriever = st.session_state.db.as_retriever()
            response = get_response(retriever, query)
            st.write(response)
