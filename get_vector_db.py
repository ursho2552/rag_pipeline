import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = os.getenv('CHROMA_PATH', './chroma')  # Ensure persistence in a relative directory
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'local-rag')  # Use a meaningful default

# Update embedding model initialization
TEXT_EMBEDDING_MODEL = os.getenv('TEXT_EMBEDDING_MODEL', 'nomic-embed-text')

def get_vector_db():
    embedding = OllamaEmbeddings(model=TEXT_EMBEDDING_MODEL)  # Make embedding model dynamic via environment
    db = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,  # Ensure database is persisted
        embedding_function=embedding
    )
    return db

