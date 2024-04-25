PROJECT_NAME:str = "RagFull project"


import ragtime
from ragtime import expe, generators
from ragtime.expe import Expe
from RAG import load_env, load_documents, create_index
from classes import MyRetriever, MyAnswerPptr
from ragtime.expe import Expe
from pathlib import Path
from llama_index.core.retrievers import VectorIndexRetriever
from ragtime.generators import StartFrom, PptrFactsFRv2, PptrEvalFRv2


# always start with init_project before importing ragtime.config values since they are updated
# with init_project and import works by value and not by reference, so values imported before
# calling init_project are not updated after the function call
ragtime.config.init_project(name=PROJECT_NAME, init_type="globals_only")
from ragtime.config import FOLDER_ANSWERS, FOLDER_QUESTIONS, logger, FOLDER_SST_TEMPLATES, FOLDER_FACTS, FOLDER_EVALS
ragtime.config.init_win_env(['OPENAI_API_KEY'])


load_env()
folders=["data1", "data2"]
documents = load_documents(folders)
index = create_index(documents)

# retrieve the top 10 most similar nodes using embeddings
retriever = VectorIndexRetriever(index=index, similarity_top_k=10)

my_retriever = MyRetriever(vector_retriever=retriever)


logger.debug('MAIN STARTS')

# generators.gen_Answers(folder_in=FOLDER_QUESTIONS, folder_out=FOLDER_ANSWERS,
#                         json_file='questions--16Q_64C_0F_0M_0A_0HE_0AE_2024-04-25_14h25,56.json',
#                         prompter=MyAnswerPptr(), b_missing_only=True,
#                         llm_names=["gpt-3.5-turbo"], retriever= my_retriever)

expe.export_to_html(json_path=FOLDER_ANSWERS / 'questions--16Q_64C_0F_1M_16A_0HE_0AE_2024-04-25_15h27,37.json')
expe.export_to_spreadsheet(json_path=FOLDER_ANSWERS / "questions--16Q_64C_0F_1M_16A_0HE_0AE_2024-04-25_15h27,37.json",
                           template_path=FOLDER_SST_TEMPLATES/'spreadsheet_rich_template.xlsx')


logger.debug('MAIN ENDS')

