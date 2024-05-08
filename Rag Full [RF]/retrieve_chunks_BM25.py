import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.retrievers.bm25 import BM25Retriever

from Rag import nodes_cr, read_doc
from ragtime.expe import Expe
from pathlib import Path
from classes import MyRetriever2

def main(path : Path) :

    exper = Expe('expe/01. Questions/questions--30Q_0C_0F_0M_0A_0HE_0AE_2024-04-24_13h47,06.json')
    nodes = nodes_cr(name=path)

    # retireve the top 10 most similar nodes using bm25
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)

    hybrid_retriever = MyRetriever2(
        bm25_retriever=bm25_retriever
    )
    for i in range(len(exper)):
        hybrid_retriever.retrieve(exper[i])
    exper.save_to_json(path = Path("expe/01. Questions/questions.json"))

if __name__ == '__main__':
    path = "pdf/docs/Conditions générales"
    main(path=path)