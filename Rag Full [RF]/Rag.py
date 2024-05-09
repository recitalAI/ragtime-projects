from html.entities import name2codepoint
import logging
import pickle
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
    load_index_from_storage,
)
from llama_index.core.evaluation import DatasetGenerator, RelevancyEvaluator
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
import random
from ragtime.expe import Expe
from pathlib import Path
import hashlib

def list_files(directory):
    files_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            files_list.append(f"{root}/{file}")
    return files_list

def generate_unique_name(directory, Dir = "storage"):
    # Concatenate file names
    files_list = list_files(directory)
    concatenated_names = ''.join([os.path.basename(file_path) for file_path in files_list])
    # Generate hash from concatenated names
    unique_name = hashlib.md5(concatenated_names.encode()).hexdigest()
    # Combine directory path with unique name
    storage_directory = os.path.join(Dir, unique_name)
    return storage_directory

def read_doc(name:str, recursive=True):
    return SimpleDirectoryReader(input_files=list_files(name), exclude_hidden=False, recursive=recursive).load_data()


def question_gen(name:str, recursive=True):

    documents = read_doc(name, recursive=recursive)

    data_generator = DatasetGenerator.from_documents(documents)
    eval_questions = data_generator.generate_questions_from_nodes()
    random.shuffle(eval_questions)
    return eval_questions

def nodes_cr(name:str, recursive=True):
    # Generate unique name for storage directory
    storage_dir = generate_unique_name(name,Dir = "Nodes")
    
    # Check if storage directory exists, create it if not
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)

    # Check if nodes file already exists
    nodes_file = os.path.join(storage_dir, "nodes.pkl")
    if os.path.exists(nodes_file):
        # Load nodes from file
        with open(nodes_file, "rb") as f:
            nodes = pickle.load(f)
    else:
        documents = read_doc(name, recursive=recursive)
        llm = OpenAI(model="gpt-3.5-turbo")
        # Initialize storage context (by default it's in-memory)
        splitter = SentenceSplitter(chunk_size=256)
        # Create nodes
        nodes = splitter.get_nodes_from_documents(documents)
        
        # Save nodes to file
        with open(nodes_file, "wb") as f:
            pickle.dump(nodes, f)

    return nodes
        
def Node_page(nodes: list, nodes_ext: dict, all_nodes: list) -> dict:
    nodes_info = {}
    for r in nodes:
        for ex in all_nodes :
            if ex.node_id == r.id_:
                nodes_info[r.id_] = {
                    "text": ex.text,
                    "score": ex.score,
                    "Node id": ex.node_id,
                    "display_name": r.metadata.get("file_name"),
                    "page_number": r.metadata.get("page_label")
                }
    if len(all_nodes) != len(nodes_info) :
        nodes_info.update(Node_page_extra(nodes_ext, all_nodes))
    return nodes_info

def Node_page_extra(nodes: dict, all_nodes: list) -> dict:
    nodes_info = {}
    for idx, val in nodes.items():
        for ex in all_nodes :
            if ex.node_id == idx:
                nodes_info[idx] = {
                    "text": ex.text,
                    "score": ex.score,
                    "Node id": ex.node_id,
                    "display_name": val.metadata.get("file_name"),
                    "page_number": val.metadata.get("page_label")
                }
    return nodes_info

def index_cr(nodes, name : str) :
    # check if storage already exists
    PERSIST_DIR = generate_unique_name(directory=name)
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