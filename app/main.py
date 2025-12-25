from retriever.rag_chain import load_qa_chain

qa = load_qa_chain()

query = "What are COVID-19 symptoms visible in chest X-rays?"
result = qa(query)

print("\n🩺 Answer:\n", result["result"])
