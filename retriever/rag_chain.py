from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from llm.llm_loader import load_llm

def load_qa_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory="vectorstore/chroma_db",
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    llm = load_llm()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa
