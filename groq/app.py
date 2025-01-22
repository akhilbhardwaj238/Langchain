import streamlit as st
import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="llama3.2")
    
    # Load documents
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()
    
    # Split documents for vectorization
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    
    # Create a vector store with FAISS
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("ChatGroq Demo")
llm_groq = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

# Define the prompt
prompt_template = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions: {input}
""")

# Create document chain
document_chain = create_stuff_documents_chain(llm_groq, prompt_template)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# User input
prompt = st.text_input("Input your prompt here")

if prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    st.write(f"Response time: {time.process_time() - start}")
    st.write(response['answer'])
    
    # Show document similarity search results
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
