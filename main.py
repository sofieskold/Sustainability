import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# Configuration
st.set_page_config(page_title="ABB | Sustainability AI", page_icon="ABB", layout="wide")

# ABB styling
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    .stButton>button { background-color: #ff0000; color: white; border: none; font-weight: bold; height: 3em; }
    .stInfo { border-left: 5px solid #ff0000; background-color: #f9f9f9; color: #333; }
    </style>
    """, unsafe_allow_html=True)

# Setup
# This function loads the PDF, splits it into chunks, and creates a vector store
# @st.cache_resource means it only runs once and is cached, which keeps the app fast
@st.cache_resource
def get_retriever():
    if not os.path.exists("faiss_index"):
        # Load and split the PDF into chunks
        loader = PyPDFLoader("rapport.pdf")
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        
        # Create embeddings and save the vector store locally
        embeddings = OllamaEmbeddings(model="llama3.2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local("faiss_index")
    else:
        # Load the existing vector store
        embeddings = OllamaEmbeddings(model="llama3.2")
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Return a retriever that fetches the 3 most relevant chunks
    return vectorstore.as_retriever(search_kwargs={"k": 3})

@st.cache_resource
def get_llm():
    return ChatOllama(model="llama3.2", temperature=0)

# Initialize retriever and LLM
retriever = get_retriever()
llm = get_llm()

# UI
col_logo, col_title = st.columns([1, 10])
with col_logo:
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/00/ABB_logo.svg", width=80)
with col_title:
    st.title("Corporate Sustainability AI Agent")
    st.caption("Supply Chain Risk Analysis")

st.markdown("---")

# Query
with st.form("query_form"):
    query = st.text_input("Enter your query:", placeholder="e.g., What are our main exposure risks in cobalt sourcing?")
    submitted = st.form_submit_button("Run Analysis")

if submitted:
    if query:
        st.markdown(f"**Query:** {query}")
        
        # Step 1: Retrieve relevant chunks from the document
        with st.status("Searching document...", expanded=False):
            context_docs = retriever.invoke(query)
            context_text = "\n\n".join([doc.page_content for doc in context_docs])

        # Step 2: Build the prompt with context and question
        prompt = ChatPromptTemplate.from_template("""
        You are a Senior ABB Sustainability Expert.

        Based ONLY on the following context, provide a professional risk assessment.
        If the context does not contain enough information to answer the question, 
        respond with: "This information is not available in the provided document."
        Do not use any knowledge outside of the context below.

        Context: {context}

        Question: {question}

        Format your response with clear bullet points.
        """)

        # Step 3: Stream the response
        st.markdown("#### Analysis")
        placeholder = st.empty()
        full_response = ""

        chain = prompt | llm | StrOutputParser()

        for chunk in chain.stream({"context": context_text, "question": query}):
            full_response += chunk
            placeholder.info(full_response + "▌")

        placeholder.info(full_response)
    else:
        st.warning("Please enter a query first.")