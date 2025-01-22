import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

# Load API keys
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv('GROQ_API_KEY')

#st.set_page_config(page_title="Chatbot Demo", layout="centered", page_icon="💬")
st.title("RAG powered with Llama3 ")


llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.2-3b-preview")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

def vector_embedding(uploaded_file):
    # Reset the vector store and related session states to handle new files
    st.session_state.clear()
    
    temp_file_path = f"./temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.session_state.embeddings = OpenAIEmbeddings()
    st.session_state.loader = PyPDFLoader(temp_file_path)
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

    os.remove(temp_file_path)  # Clean up

uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

if uploaded_file and st.button("Process Document"):
    vector_embedding(uploaded_file)
    st.success("Document processed and vector store created!")

if "vectors" in st.session_state:
    prompt1 = st.text_input("Ask your question:")
    if prompt1:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write(f"Response time: {time.process_time() - start:.2f} seconds")
        st.write("**Answer:**", response['answer'])

        with st.expander("Document Similarity Search"):
            for doc in response["context"]:
                st.write(doc.page_content)
                st.write("--------------------------------")
