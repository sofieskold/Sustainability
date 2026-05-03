import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# --- KONFIGURATION ---
st.set_page_config(page_title="ABB | Sustainability AI", page_icon="ABB", layout="wide")

# ABB Brand Colors & Styling
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    .stButton>button { background-color: #ff0000; color: white; border: none; font-weight: bold; height: 3em; }
    .stInfo { border-left: 5px solid #ff0000; background-color: #f9f9f9; color: #333; }
    </style>
    """, unsafe_allow_html=True)

# --- SNABBARE AI-SETUP ---
@st.cache_resource
def get_retriever():
    if not os.path.exists("faiss_index"):
        loader = PyPDFLoader("rapport.pdf")
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        embeddings = OllamaEmbeddings(model="llama3.2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local("faiss_index")
    else:
        embeddings = OllamaEmbeddings(model="llama3.2")
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

@st.cache_resource
def get_llm():
    return ChatOllama(model="llama3.2", temperature=0)

# --- INITIALISERING ---
retriever = get_retriever()
llm = get_llm()

# --- UI ---
col_logo, col_title = st.columns([1, 10])
with col_logo:
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/00/ABB_logo.svg", width=80)
with col_title:
    st.title("Corporate Sustainability AI Agent")
    st.caption("Agentic AI Innovation for Supply Chain Risk Management")

st.markdown("---")

with st.sidebar:
    st.header("System Status")
    st.success("Local LLM: Llama 3.2 Active")
    st.caption("Data Privacy: Local & Secure")
    st.divider()
    
    st.markdown("**Risk Dimensions**")
    st.info("Select focus areas for analysis:")
    
    # Utökade dimensioner baserat på ABB:s jobbannons
    hr_active = st.checkbox("Human Rights", value=True, help="Analyze risks related to social impact and rights.")
    labor_active = st.checkbox("Labor Practices", value=True, help="Focus on working conditions and fair labor.")
    climate_active = st.checkbox("Climate Impacts", help="Identify carbon footprint and climate-related risks.")
    wu_active = st.checkbox("Water Usage", help="Map water scarcity and usage efficiency.")
    bio_active = st.checkbox("Biodiversity", help="Assess impact on local ecosystems and species.")

# --- ANALYS-LOGIK ---
query = st.text_input("Enter your query:", placeholder="e.g., What are our main exposure risks in cobalt sourcing?")

if st.button("Run Risk Analysis"):
    if query:
        # Samla alla valda dimensioner i en lista
        active_dims = []
        if hr_active: active_dims.append("Human Rights")
        if labor_active: active_dims.append("Labor Practices")
        if climate_active: active_dims.append("Climate Impacts")
        if wu_active: active_dims.append("Water Usage")
        if bio_active: active_dims.append("Biodiversity")
        
        dim_string = ", ".join(active_dims) if active_dims else "General Sustainability"

        # Hämta dokument
        with st.status("Agent searching value chain data...", expanded=False):
            context_docs = retriever.invoke(query)
            context_text = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Uppdaterad prompt som styr AI:n mot valda dimensioner
        prompt = ChatPromptTemplate.from_template("""
        You are a Senior ABB Sustainability Expert. 
        Focus your analysis specifically on these dimensions: {dimensions}.
        
        Based ONLY on the following context, provide a professional risk assessment:
        Context: {context}
        
        Question: {question}
        
        Format your response with clear bullet points.
        """)
        
        st.markdown(f"#### Agent Insights: {dim_string}")
        placeholder = st.empty()
        full_response = ""
        
        chain = prompt | llm | StrOutputParser()
        
        # Streamad respons
        for chunk in chain.stream({
            "context": context_text, 
            "question": query, 
            "dimensions": dim_string
        }):
            full_response += chunk
            placeholder.info(full_response + "▌")
        
        placeholder.info(full_response)
    else:
        st.warning("Please enter a query first.")