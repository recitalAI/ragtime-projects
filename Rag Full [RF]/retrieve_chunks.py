import logging
import sys
from unittest.util import strclass

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.retrievers.bm25 import BM25Retriever

from Rag import nodes_cr, index_cr
from ragtime.expe import Expe
from pathlib import Path
from classes import MyRetriever
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
def main(path : strclass) :

    exper = Expe('expe/01. Questions/questions--30Q_0C_0F_0M_0A_0HE_0AE_2024-04-24_13h47,06.json')
    nodes = nodes_cr(name=path)
    index = index_cr(nodes)

    # retireve the top 10 most similar nodes using embeddings
    vector_retriever = index.as_retriever(similarity_top_k=10)

    # retireve the top 10 most similar nodes using bm25
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)

    hybrid_retriever = MyRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever
    )
    for i in range(len(exper)):
        hybrid_retriever.retrieve(exper[i])
    exper.save_to_json(path = Path("expe/01. Questions/questions.json"))


if __name__ == '__main__':
    path = "pdf/docs/Conditions générales"
    main(path=path)