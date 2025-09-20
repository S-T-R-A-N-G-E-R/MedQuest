import streamlit as st
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM # <-- UPDATED IMPORT
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- 1. SETUP LOGGING & PAGE CONFIG ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
st.set_page_config(page_title="MedQuest", page_icon="⚕️", layout="wide")
st.title("⚕️ MedQuest: Your Medical Research Assistant")

# --- 2. DEFINE CONSTANTS ---
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3:8b"

# --- 3. DEFINE THE PROMPT TEMPLATE ---
custom_prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Please provide a concise and direct answer based *only* on the provided context.

Context: {context}

Question: {question}
Helpful Answer:"""

# --- 4. CACHED FUNCTION TO LOAD THE RAG CHAIN ---
@st.cache_resource
def load_chain():
    try:
        logging.info("Loading embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logging.info(f"Loading FAISS vector store from {DB_FAISS_PATH}...")
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        
        # --- ** THIS IS THE FIX ** ---
        # Switched to a more reliable retriever that always returns documents.
        retriever = db.as_retriever(search_kwargs={"k": 3}) 
        
        logging.info(f"Initializing Ollama with model {LLM_MODEL}...")
        # --- ** FIXES DEPRECATION WARNING ** ---
        llm = OllamaLLM(model=LLM_MODEL)
        
        prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

        # This function formats the retrieved documents into a single string.
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        logging.info("RAG chain loaded successfully.")
        return rag_chain, retriever

    except Exception as e:
        logging.error(f"Failed to load RAG chain: {e}", exc_info=True)
        st.error(f"Failed to load the RAG chain. Please check the logs. Error: {e}")
        return None, None

# --- 5. INITIALIZE SESSION STATE & LOAD CHAIN ---
# (Rest of the file is the same, but included for completeness)
if "messages" not in st.session_state:
    st.session_state.messages = []

chain, retriever = load_chain()

# --- 6. DISPLAY CHAT HISTORY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 7. HANDLE USER INPUT ---
if prompt := st.chat_input("Ask a question about your medical data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if chain is None or retriever is None:
                st.error("The application is not configured correctly. Please check the server logs.")
            else:
                try:
                    response = chain.invoke(prompt)
                    st.markdown(response)
                    
                    # We can still show the sources, just without a score.
                    retrieved_docs = retriever.invoke(prompt)
                    if retrieved_docs:
                        with st.expander("Show Retrieved Sources"):
                            for doc in retrieved_docs:
                                source = doc.metadata.get('source', 'Unknown Source')
                                st.info(f"**Source:** `{source}`")
                                st.caption(f"Content: {doc.page_content[:250]}...")
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    logging.error(f"Error during query processing: {e}", exc_info=True)
                    st.error("An error occurred while processing your question. Please try again.")

