import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Response
from llama_index.core.retrievers import VectorIndexRetriever
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
    os.environ['OPENAI_API_KEY']= ''


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
