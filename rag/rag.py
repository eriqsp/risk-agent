from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = PROJECT_ROOT / "documents" / "risk_policy.pdf"
DB_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "risk_policy"


def create_vector_store():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_PATH),
        collection_name=COLLECTION_NAME,
    )

    return vector_store


def load_vector_store():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    return Chroma(
        persist_directory=str(DB_PATH),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )


def search_policy(query: str, k: int = 3):  # k = 3 gives the agent enough surrounding policy information
    vector_store = load_vector_store()
    return vector_store.similarity_search(query, k=k)
