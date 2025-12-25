from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

import os

PDF_PATH = "data/raw_pdfs/who_covid_guidelines.pdf"
DB_DIR = "vectordb/chroma"

# 1. Load PDF
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# 2. Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# 3. Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Store in ChromaDB
vectorstore = Chroma.from_documents(
    docs,
    embedding_model,
    persist_directory=DB_DIR
)

vectorstore.persist()

print("✅ Embeddings created & stored in ChromaDB")
