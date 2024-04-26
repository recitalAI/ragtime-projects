import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from RAG import load_env, load_documents, create_index
from classes import MyRetriever
from llama_index.core.retrievers import VectorIndexRetriever
from ragtime.expe import Expe
from pathlib import Path

load_env()

result = Expe('expe/01. Questions/questionsTest(English).json')

folders=["data1", "data2"]
documents = load_documents(folders)
index = create_index(documents)

# retrieve the top 10 most similar nodes 
retriever =  VectorIndexRetriever(index=index, similarity_top_k=4)

my_retriever = MyRetriever(vector_retriever=retriever)


for i in range(len(result)):
    my_retriever.retrieve(result[i])
exper.save_to_json(path = Path("expe/01. Questions/questions.json"))
