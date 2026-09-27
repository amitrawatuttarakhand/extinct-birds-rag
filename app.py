import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Fetch API key from Streamlit Secrets or Environment Variables
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

@st.cache_resource(show_spinner="Initializing Green Vector Database...")
def init_vector_store():
    loader = TextLoader("extinct_birds_data.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    # Low-energy CPU embedding model (~130MB)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return Chroma.from_documents(docs, embeddings)

@st.cache_resource
def init_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY, 
        model_name="llama-3.2-1b-preview",  # Extremely lightweight and fast
        temperature=0.2
    )

vectorstore = init_vector_store()
llm = init_llm()
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
