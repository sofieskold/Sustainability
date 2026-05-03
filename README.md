# Sustainability
Sustainability AI Agent
A local RAG-based AI agent for sustainability document analysis and supply chain risk assessment, built with LangChain, Ollama, and Streamlit.

What it does
Upload a sustainability report (PDF) and query it using natural language. The agent retrieves relevant sections from the document and generates a professional risk assessment, without hallucinating information that isn't there.
Built as a prototype for automating sustainability workflows, inspired by ABB's work in supply chain risk management.

How it works
The PDF is split into chunks and stored in a local FAISS vector store
When a query is submitted, the most relevant chunks are retrieved
A local LLM (Llama 3.2 via Ollama) generates a response based only on the retrieved context
If the answer is not in the document, the agent says so

Tech stack
LangChain — RAG pipeline
Ollama — local LLM (Llama 3.2)
FAISS — vector store
Streamlit — UI

Installation

Install Ollama and pull the model:

bashollama pull llama3.2

Add your PDF report and name it rapport.pdf in the project folder
Run the app:

bashstreamlit run sustainability_agent.py
