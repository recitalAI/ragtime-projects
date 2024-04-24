PROJECT_NAME:str = "Pdf_QA_tester"

import ragtime
from ragtime import expe, generators
from ragtime.generators import StartFrom, PptrFactsFRv2, PptrSimpleEvalFR

from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# always start with init_project before importing ragtime.config values since they are updated
# with init_project and import works by value and not by reference, so values imported before
# calling init_project are not updated after the function call
ragtime.config.init_project(name=PROJECT_NAME, init_type="globals_only")
from ragtime.config import FOLDER_ANSWERS, FOLDER_FACTS, FOLDER_EVALS, logger, FOLDER_SST_TEMPLATES
ragtime.config.init_win_env(['OPENAI_API_KEY', 'GOOGLE_API_KEY'])



logger.debug('MAIN STARTS')

#generators.gen_Facts(folder_in=FOLDER_ANSWERS, folder_out=FOLDER_FACTS, json_file='questions--30Q_600C_0F_2M_60A_0HE_0AE_2024-04-24_14h10,25.json',
#                     llm_names=['gpt-4'], prompter=PptrFactsFRv2())

#generators.gen_Evals(folder_in=FOLDER_FACTS, folder_out=FOLDER_EVALS, 
#                     json_file='questions--30Q_600C_174F_2M_60A_60HE_0AE_2024-04-24_14h25,44.json',
#                     llm_names=['gpt-4'], prompter=PptrSimpleEvalFR())


expe.export_to_html(json_path=FOLDER_EVALS / "questions--30Q_600C_174F_2M_60A_60HE_59AE_2024-04-24_14h28,59.json")
expe.export_to_spreadsheet(json_path=FOLDER_EVALS / "questions--30Q_600C_174F_2M_60A_60HE_59AE_2024-04-24_14h28,59.json",
                           template_path=FOLDER_SST_TEMPLATES/'spreadsheet_rich_template.xlsx')


logger.debug('MAIN ENDS')
# Note: the logger can be used only *after* ragtime.config.init_project
logger.debug(f'{PROJECT_NAME} STARTS')