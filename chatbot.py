import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- 1. SETUP LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 2. DEFINE CONSTANTS ---
# Must be the same as in run_pipeline.py
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# The model we are running with Ollama
LLM_MODEL = "llama3:8b"

# --- 3. DEFINE THE PROMPT TEMPLATE ---
# This is the heart of the RAG system.
# It instructs the LLM how to use the retrieved context.
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# --- 4. CREATE THE RAG CHAIN ---
def create_qa_chain():
    """
    Creates and returns a RetrievalQA chain.
    """
    try:
        # Load the embeddings model
        logging.info("Loading embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Load the FAISS vector store
        logging.info(f"Loading FAISS vector store from {DB_FAISS_PATH}...")
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)


        # Initialize the Ollama LLM
        logging.info(f"Initializing Ollama with model {LLM_MODEL}...")
        llm = Ollama(model=LLM_MODEL)

        # Create a prompt from the template
        prompt = PromptTemplate(
            template=custom_prompt_template, input_variables=["context", "question"]
        )

        # Create the RetrievalQA chain
        logging.info("Creating RetrievalQA chain...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3}), # Retrieve top 3 documents
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
        logging.info("Chain created successfully.")
        return qa_chain

    except Exception as e:
        logging.error(f"Failed to create QA chain: {e}", exc_info=True)
        return None

# --- 5. RUN THE CHATBOT ---
def main():
    qa_chain = create_qa_chain()
    if not qa_chain:
        print("Exiting due to chain creation failure.")
        return

    print("\n--- MedQuest Chatbot ---")
    print("Ask a question about your medical research data. Type 'quit' to exit.")
    
    while True:
        query = input("\nYour Question: ")
        if query.lower() == "quit":
            break
        
        print("Searching for answers...")
        try:
            # Run the chain
            result = qa_chain.invoke({"query": query})
            
            # Print the answer
            print("\n--- Answer ---")
            print(result["result"])

            # Print the sources
            print("\n--- Sources ---")
            if result.get("source_documents"):
                for doc in result["source_documents"]:
                    # We get the source from the metadata
                    source = doc.metadata.get('source', 'Unknown')
                    print(f"- {source}")
            else:
                print("No sources found.")

        except Exception as e:
            logging.error(f"An error occurred during query processing: {e}", exc_info=True)
            print("Sorry, an error occurred. Please try again.")

if __name__ == "__main__":
    main()
