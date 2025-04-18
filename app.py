import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from PyPDF2 import PdfReader

# Fetch Hugging Face API token securely from Streamlit Secrets
hf_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# Set your Hugging Face API token in the environment variable
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# Title of the Streamlit app
st.title("Chat with your PDF")

# File uploader in Streamlit for PDF files
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file is not None:
    # Read PDF content
    pdf_reader = PdfReader(uploaded_file)
    text = "".join([page.extract_text() for page in pdf_reader.pages])

    # Extract embeddings using Hugging Face Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_texts([text], embeddings)

    # Set up the Retriever
    retriever = vectordb.as_retriever()

    # Initialize Hugging Face Hub LLM (e.g., FLAN-T5)
    llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.7, "max_length": 512})

    # Create Retrieval-based QA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    # Input text box for user queries
    query = st.text_input("Ask a question about your PDF:")
    if query:
        # Generate answer
        result = qa_chain({"query": query})
        answer = result["result"]
        source_documents = result["source_documents"]

        # Display answer
        st.write("Answer:", answer)

        # Display source documents
        with st.expander("Source Documents"):
            for i, doc in enumerate(source_documents):
                st.write(f"Source {i + 1}:")
                st.write(doc.page_content)
