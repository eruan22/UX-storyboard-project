import os

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from models.schemas import Panel

# load and chunk pdf documents
def load_pdfs(pdf_dir: str):
    """load and chunk PDFs from a folder"""
    loader = DirectoryLoader(
        pdf_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    raw_docs = loader.load()

    # Split into chunks — important for retrieval accuracy
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # ~a paragraph
        chunk_overlap=50,     # overlap so context isn't cut off
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"Loaded {len(raw_docs)} pages → {len(chunks)} chunks")
    return chunks

# call pdfs and load into ChromaDB
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# initialize ChromaDB client
CHROMA_PATH = "./data/chroma_db"
COLLECTION = "ux-research-docs"

# vector store function
def get_vectorstore():
    """Initialize ChromaDB vector store with Ollama embeddings."""
    # get embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # reduce runtime if already loaded
    if os.path.exists(CHROMA_PATH):
        print("Loading existing ChromaDB...")
        return Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings,
            persist_directory=CHROMA_PATH
        )
    # load and chunk your PDFs
    chunks = load_pdfs("UX-documents")

    # build vector store with ChromaDB
    vector_store = Chroma(
        collection_name = COLLECTION,
        embedding_function = embeddings,
        persist_directory = CHROMA_PATH
    )

    vector_store.add_documents(chunks)
    return vector_store
# get embeddings
# embeddings = OllamaEmbeddings(model="nomic-embed-text")


# # build vector store with ChromaDB
# vector_store = Chroma(
#     collection_name = COLLECTION,
#     embedding_function = embeddings,
#     persist_directory = CHROMA_PATH
# )

# vector_store.add_documents(chunks)

# BASIC RETRIEVE FUNCTION
def basic_retrieve(panels: List[Panel], vector_store, top_k: int) -> List[str]:
    """Retrieve the top k most similar documents for a query.

    Args:
        panels: List of UI panels to search for
        top_k: Number of documents to retrieve

    Returns:
        List of dictionaries with 'content', 'metadata', and 'score' keys"""
    # initialize retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    # combine panel content into a single query
    query = " ".join([f"{panel.action} {panel.context}" for panel in panels])
    # retrieve top 5 documents
    docs = retriever.invoke(query)
    return [doc.page_content for doc in docs]

