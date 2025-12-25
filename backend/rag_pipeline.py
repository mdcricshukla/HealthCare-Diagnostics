from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

DB_DIR = "vectordb/chroma"

# Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector DB
vector_db = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embedding_model
)

# FLAN-T5 pipeline
hf_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever()
)

def get_diagnosis(query: str):
    return qa_chain.run(query)
