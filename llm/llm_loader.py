from langchain.llms import HuggingFacePipeline
from transformers import pipeline

def load_llm():
    pipe = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_length=256
    )
    return HuggingFacePipeline(pipeline=pipe)
