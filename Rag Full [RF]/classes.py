from importlib.metadata import metadata
from typing import Optional
from llama_index.core.retrievers import BaseRetriever
from ragtime.expe import QA, Chunks, Prompt, Question, WithLLMAnswer, Chunk, Answer
from ragtime.generators import Retriever, Prompter
from typing import Optional, TypeVar, Union
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever
from langdetect import detect
from unidecode import unidecode
from ragtime.config import logger



class MyRetriever(Retriever, BaseRetriever):
    vector_retriever: VectorIndexRetriever
    bm25_retriever: BM25Retriever
        
        
    class Config:
        arbitrary_types_allowed = True

    def _retrieve(self, query: str, indexer=None, similarity_top_k: Optional[int] = None, **kwargs) -> list:
        bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)

        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes

    def retrieve(self, qa: QA, **kwargs):
        result = self._retrieve(qa.question.text, **kwargs)
        for r in result:
            chunk = Chunk()
            chunk.text , chunk.meta = r.text , r.meta
            # Check if the chunk already exists in qa.chunks
            existing_chunk = next((c for c in qa.chunks if c.text == chunk.text and c.meta == chunk.meta), None)
            if existing_chunk is None:
                qa.chunks.append(chunk)


class MyRetriever2(Retriever, BaseRetriever):
    bm25_retriever: BM25Retriever
        
        
    class Config:
        arbitrary_types_allowed = True

    def _retrieve(self, query: str, indexer=None, similarity_top_k: Optional[int] = None, **kwargs) -> list:
        bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in bm25_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes

    def retrieve(self, qa: QA, **kwargs):
        result = self._retrieve(qa.question.text, **kwargs)
        for r in result:
            chunk = Chunk()
            chunk.text , chunk.meta = r.text , {"score":r.score, "Node id" : r.node_id, "display_name" :r.node.metadata["file_name"], "page_number": None }
            # Check if the chunk already exists in qa.chunks
            existing_chunk = next((c for c in qa.chunks if c.text == chunk.text and c.meta == chunk.meta), None)
            if existing_chunk is None:
                qa.chunks.append(chunk)