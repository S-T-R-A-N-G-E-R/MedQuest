import os
import logging
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
    JSONLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. SETUP LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 2. DEFINE CONSTANTS ---
DATA_PATH = "./data"
DB_FAISS_PATH = "vectorstore/db_faiss"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
# --- **SWITCH TO A FASTER MODEL** ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- 3. SETUP DOCUMENT LOADERS (Unchanged) ---
LOADER_MAPPING = {
    ".csv": (CSVLoader, {"encoding": "utf-8"}),
    ".json": (JSONLoader, {"jq_schema": ".", "text_content": False}),
    ".txt": (TextLoader, {"encoding": "utf-8"}),
    ".pdf": (PyPDFLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
}

# --- 4. CREATE THE LOADING FUNCTION (Unchanged) ---
def load_documents(source_dir: str) -> list[Document]:
    all_docs = []
    logging.info(f"Loading documents from: {source_dir}")
    if not os.path.exists(source_dir):
        logging.error(f"Source directory '{source_dir}' not found. Aborting.")
        return []
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_ext = os.path.splitext(file_name)[1].lower()
            if file_ext in LOADER_MAPPING:
                loader_class, loader_args = LOADER_MAPPING[file_ext]
                logging.info(f"Loading '{file_path}' with {loader_class.__name__}...")
                try:
                    loader = loader_class(file_path, **loader_args)
                    docs = loader.load()
                    all_docs.extend(docs)
                except Exception as e:
                    logging.error(f"Error loading '{file_path}': {e}", exc_info=True)
            else:
                logging.warning(f"Skipping '{file_path}': unsupported file type '{file_ext}'")
    return all_docs

# --- 5. CREATE THE CHUNKING FUNCTION (Unchanged) ---
def process_documents(documents: list[Document]) -> list[Document]:
    if not documents:
        logging.warning("No documents to process. Returning an empty list.")
        return []
    logging.info(f"Splitting {len(documents)} documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)
    logging.info(f"Created {len(texts)} text chunks.")
    return texts

# --- 6. CREATE THE VECTOR STORE FUNCTION (Unchanged logic, but will use faster model) ---
def create_vector_store(texts: list[Document]):
    if not texts:
        logging.warning("No text chunks to process. Skipping vector store creation.")
        return
    logging.info(f"Initializing HuggingFaceEmbeddings with model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logging.info("Creating FAISS vector store... This will be much faster now.")
    db = FAISS.from_documents(texts, embeddings)
    logging.info(f"Saving vector store to: {DB_FAISS_PATH}")
    db.save_local(DB_FAISS_PATH)
    logging.info("Vector store created and saved successfully.")

# --- 7. RUN THE SCRIPT ---
if __name__ == "__main__":
    # --- **NEW: CHECK IF VECTOR STORE ALREADY EXISTS** ---
    if os.path.exists(DB_FAISS_PATH):
        logging.info("Vector store already exists. Skipping the creation process.")
    else:
        logging.info("Vector store not found. Starting the creation process.")
        # Step 1: Load documents
        documents = load_documents(DATA_PATH)
        # Step 2: Chunk documents
        texts = process_documents(documents)
        # Step 3: Create vector store
        create_vector_store(texts)

    logging.info("--- All steps completed successfully! ---")

