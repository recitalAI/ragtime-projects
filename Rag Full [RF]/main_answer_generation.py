PROJECT_NAME:str = "Pdf_QA_tester"

import ragtime
from ragtime import expe, generators
from classes import MyRetriever2, MCQAnsPptr
from Rag import nodes_cr
from dotenv import load_dotenv
from llama_index.retrievers.bm25 import BM25Retriever
import os

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
# always start with init_project before importing ragtime.config values since they are updated
# with init_project and import works by value and not by reference, so values imported before
# calling init_project are not updated after the function call
ragtime.config.init_project(name=PROJECT_NAME, init_type="globals_only")
from ragtime.config import FOLDER_ANSWERS, FOLDER_QUESTIONS, FOLDER_FACTS, FOLDER_EVALS, logger, FOLDER_SST_TEMPLATES
ragtime.config.init_win_env(['GEMINI_API_KEY', 'ANTHROPIC_API_KEY'])
#nodes = nodes_cr(name="pdf/docs/Conditions générales")

# retireve the top 10 most similar nodes using bm25
#bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)

logger.debug('MAIN STARTS')

#generators.gen_Answers(folder_in=FOLDER_QUESTIONS, folder_out=FOLDER_ANSWERS,
#                       json_file='questions--30Q_300C_0F_0M_0A_0HE_0AE_2024-05-06_17h12,56.json',
#                        prompter=MCQAnsPptr(), b_missing_only=True,
#                        llm_names=["gemini/gemini-pro", "claude-3-opus-20240229"],retriever = MyRetriever2(bm25_retriever=bm25_retriever))

expe.export_to_html(json_path=FOLDER_ANSWERS / 'questions--30Q_300C_0F_2M_59A_0HE_0AE_2024-05-06_17h38,19.json')
expe.export_to_spreadsheet(json_path=FOLDER_ANSWERS / "questions--30Q_300C_0F_2M_59A_0HE_0AE_2024-05-06_17h38,19.json",
                           template_path=FOLDER_SST_TEMPLATES/'spreadsheet_rich_template.xlsx')


logger.debug('MAIN ENDS')
# Note: the logger can be used only *after* ragtime.config.init_project
logger.debug(f'{PROJECT_NAME} STARTS')