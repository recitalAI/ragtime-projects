import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Response
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SimilarityPostprocessor
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core.evaluation import DatasetGenerator, RelevancyEvaluator
from llama_index.llms.openai import OpenAI
import os.path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

def load_env():
    load_dotenv()
    os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')


def load_documents(folders):
    documents = []
    for folder in folders:
        documents.extend(SimpleDirectoryReader(folder).load_data())
    return documents


def create_index(documents):
    # check if storage already exists
    PERSIST_DIR = "./storage"
    if not os.path.exists(PERSIST_DIR):
        # load the documents and create the index
        documents = SimpleDirectoryReader("data1").load_data()
        documents = SimpleDirectoryReader("data2").load_data()
        index = VectorStoreIndex.from_documents(documents)
        # store it for later
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    return index


# def create_query_engine(index):
#     """
#     Creates a query engine from the index.
#     """
#     retriever = VectorIndexRetriever(index=index, similarity_top_k=4)
#     return RetrieverQueryEngine(retriever=retriever)

# def query_and_print(query_engine, query):
#     # Executes a query and prints the response.
#     response = query_engine.query(query)
#     pprint_response(response, show_source=True)
#     print(response)

def generate_questions(folders):
    for folder in folders:
        reader = SimpleDirectoryReader(folder)
    documents = reader.load_data()
    data_generator = DatasetGenerator.from_documents(documents)
    eval_questions = data_generator.generate_questions_from_nodes()
    return eval_questions




