import streamlit as st
from dotenv import load_dotenv
import os
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import shutil

# -----------------
# Load API keys
# -----------------
load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

if not cohere_api_key or not groq_api_key:
    st.error("Missing API key(s) in .env file.")
else:
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
def get_vectorstore(splits, persist_dir="./chroma_db"):
    """Cache vectorstore to avoid re-computation when same file is uploaded"""
    return Chroma.from_documents(
        splits, embedding=embedding_model, persist_directory=persist_dir
    )


def get_response(retriever, query: str):
    """RAG pipeline: retrieval + QA with longer answers"""
    system_prompt = (
        "You are a helpful assistant. Use the retrieved context to answer the question "
        "as completely as possible. If the answer is not in the context, say that you don't know, "
        "but always explain your reasoning based on the available information.\n\n{context}"
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


# -----------------
# Streamlit App
# -----------------
st.set_page_config(page_title="RAG-Utkarsh", layout="wide")
st.title("RAG Chatbot - Utkarsh")

# Session state for DB
if "db" not in st.session_state:
    st.session_state.db = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

query = st.text_input("Ask a question about the uploaded document")
submit = st.button("Answer")

with st.sidebar:
    uploaded_file = st.file_uploader("Please Upload PDF", type=["pdf"])
    if uploaded_file:
        temp_file = f"./{uploaded_file.name}"
        with open(temp_file, "wb") as file:
            file.write(uploaded_file.getvalue())

        if st.button("Get Embeddings"):
            with st.spinner("Processing..."):
                splits = load_pdf_and_split(temp_file)

                # Use separate folder per document to avoid mixing
                persist_dir = f"./chroma_db_{uploaded_file.name}"
                st.session_state.db = get_vectorstore(splits, persist_dir=persist_dir)
                st.session_state.doc_name = uploaded_file.name
                st.success(f"Embeddings for '{uploaded_file.name}' created & stored!")

    if st.button("Clear Previous Embeddings"):
        # Remove all persisted DBs
        for folder in os.listdir("."):
            if folder.startswith("chroma_db"):
                shutil.rmtree(folder)
        st.session_state.db = None
        st.session_state.doc_name = None
        st.success("All embeddings cleared!")

# Handle QA
if submit:
    if st.session_state.db is None:
        st.error("Please upload a PDF and create embeddings first.")
    else:
        with st.spinner("Responding..."):
            retriever = st.session_state.db.as_retriever()
            response = get_response(retriever, query)
            st.write(response)
