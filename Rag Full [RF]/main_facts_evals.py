PROJECT_NAME:str = "Pdf_QA_tester"

import ragtime
from ragtime import expe, generators
from ragtime.generators import StartFrom, PptrFactsFRv2, PptrSimpleEvalFR

from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
# always start with init_project before importing ragtime.config values since they are updated
# with init_project and import works by value and not by reference, so values imported before
# calling init_project are not updated after the function call
ragtime.config.init_project(name=PROJECT_NAME, init_type="globals_only")
from ragtime.config import FOLDER_ANSWERS, FOLDER_FACTS, FOLDER_EVALS, logger, FOLDER_SST_TEMPLATES
ragtime.config.init_win_env(['GEMINI_API_KEY', 'ANTHROPIC_API_KEY'])



logger.debug('MAIN STARTS')

#generators.gen_Facts(folder_in=FOLDER_ANSWERS, folder_out=FOLDER_FACTS, json_file='questions--30Q_300C_0F_2M_59A_0HE_0AE_2024-05-06_17h38,19.json',
#                     llm_names=['gemini/gemini-pro'], prompter=PptrFactsFRv2())

#generators.gen_Evals(folder_in=FOLDER_FACTS, folder_out=FOLDER_EVALS, 
#                     json_file='questions--30Q_300C_468F_2M_59A_60HE_0AE_2024-05-06_17h51,05.json',
#                     llm_names=['gemini/gemini-pro'], prompter=PptrSimpleEvalFR())


expe.export_to_html(json_path=FOLDER_EVALS / "questions--30Q_300C_468F_2M_59A_60HE_59AE_2024-05-06_18h00,21.json")
expe.export_to_spreadsheet(json_path=FOLDER_EVALS / "questions--30Q_300C_468F_2M_59A_60HE_59AE_2024-05-06_18h00,21.json",
                           template_path=FOLDER_SST_TEMPLATES/'spreadsheet_rich_template.xlsx')


logger.debug('MAIN ENDS')
# Note: the logger can be used only *after* ragtime.config.init_project
logger.debug(f'{PROJECT_NAME} STARTS')