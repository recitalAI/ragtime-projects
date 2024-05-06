import logging
import sys
import pandas as pd
import os.path

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    SimpleDirectoryReader,
    Document,
    Response,
    load_index_from_storage,
)
from llama_index.core.evaluation import DatasetGenerator, RelevancyEvaluator
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
import random
from pathlib import Path


def read_doc(name:str, recursive=True):
    return SimpleDirectoryReader(input_dir=name, recursive=recursive).load_data()


def question_gen(name:str, recursive=True):

    documents = read_doc(name, recursive=recursive)

    data_generator = DatasetGenerator.from_documents(documents)
    eval_questions = data_generator.generate_questions_from_nodes()
    random.shuffle(eval_questions)
    return eval_questions

def nodes_cr(name:str,recursive= True):
    documents = read_doc(name, recursive = recursive)
    llm = OpenAI(model="gpt-3.5-turbo")
    # initialize storage context (by default it's in-memory)
    splitter = SentenceSplitter(chunk_size=256)
    # creat nodes
    nodes = splitter.get_nodes_from_documents(
        documents
    )
    return nodes
        


def index_cr(nodes) :
    # check if storage already exists
    PERSIST_DIR = "./storage"
    if not os.path.exists(PERSIST_DIR):

        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)
        index = VectorStoreIndex(nodes, storage_context=storage_context)
        # store it for later
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)

    return index
