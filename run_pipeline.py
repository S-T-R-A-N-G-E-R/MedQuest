import os
import logging
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
    JSONLoader
)
from langchain.docstore.document import Document

# --- 1. SETUP LOGGING ---
# Use logging for better debugging and tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 2. DEFINE CONSTANTS ---
# The script expects a 'data' folder in the same directory
DATA_PATH = "./data"

# --- 3. SETUP DOCUMENT LOADERS ---
# Mapping file extensions to their corresponding loader classes and arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {"encoding": "utf-8"}),
    ".json": (JSONLoader, {"jq_schema": ".", "text_content": False}),
    ".txt": (TextLoader, {"encoding": "utf-8"}),
    ".pdf": (PyPDFLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
}

# --- 4. CREATE THE LOADING FUNCTION ---
def load_documents(source_dir: str) -> list[Document]:
    """
    Loads all documents from the specified source directory using the appropriate loader.
    """
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

# --- 5. RUN THE SCRIPT ---
if __name__ == "__main__":
    documents = load_documents(DATA_PATH)

    logging.info("--- Loading Complete ---")
    logging.info(f"Total number of documents loaded: {len(documents)}")

    if documents:
        logging.info("\n--- Sample Document ---")
        sample_doc = documents[0]
        source = sample_doc.metadata.get('source', 'N/A')
        logging.info(f"Source: {source}")
        
        # Clean up content for preview
        content_preview = ' '.join(sample_doc.page_content.splitlines())
        logging.info(f"Content preview: {content_preview[:250]}...")


