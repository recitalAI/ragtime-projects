PROJECT_NAME:str = "Rag Full [RF]"

import ragtime
from ragtime import expe, generators
from ragtime.expe import Expe, UpdateTypes
from ragtime.generators import StartFrom, PptrFactsFRv2, PptrSimpleEvalFR
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()


def update_json(json_path:Path, sheet_path: Path,update_type = UpdateTypes, data_col= int,question_col = int, answer_col = int):
  expe:Expe = Expe(json_path=json_path)
  expe.update_from_spreadsheet(path = sheet_path, update_type=update_type, data_col= data_col,question_col = question_col, answer_col = answer_col)
  expe.save_to_json(path = json_path)

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# always start with init_project before importing ragtime.config values since they are updated
# with init_project and import works by value and not by reference, so values imported before
# calling init_project are not updated after the function call
ragtime.config.init_project(name=PROJECT_NAME, init_type="globals_only")
from ragtime.config import FOLDER_ANSWERS, FOLDER_FACTS, FOLDER_EVALS, logger, FOLDER_SST_TEMPLATES
ragtime.config.init_win_env(['GEMINI_API_KEY', 'ANTHROPIC_API_KEY'])



logger.debug('MAIN STARTS')

update_json(json_path=FOLDER_ANSWERS /'questions--30Q_300C_0F_2M_58A_0HE_0AE_2024-05-08_22h56,42.json',sheet_path=FOLDER_ANSWERS /'questions--30Q_300C_0F_2M_58A_0HE_0AE_2024-05-09_13h09,42.xlsx',update_type=UpdateTypes.human_eval,data_col= 14,question_col = 1, answer_col = 8)

#generators.gen_Facts(folder_in=FOLDER_ANSWERS, folder_out=FOLDER_FACTS, json_file='questions--10Q_170C_0F_2M_20A_0HE_0AE_2024-04-22_09h26,25updated.json',
#                    llm_names=['gpt-3.5-turbo'], prompter=PptrFactsFRv2())

#generators.gen_Evals(folder_in=FOLDER_FACTS, folder_out=FOLDER_EVALS, 
#                     json_file='questions--30Q_300C_142F_2M_58A_60HE_0AE_2024-05-09_02h10,39.json',
#                     llm_names=['gpt-3.5-turbo'], prompter=PptrSimpleEvalFR())


#expe.export_to_html(json_path=FOLDER_EVALS / "questions--30Q_300C_142F_2M_58A_60HE_58AE_2024-05-09_02h23,12.json")
#expe.export_to_spreadsheet(json_path=FOLDER_EVALS / "questions--30Q_300C_142F_2M_58A_60HE_58AE_2024-05-09_02h23,12.json",
#                           template_path=FOLDER_SST_TEMPLATES/'spreadsheet_rich_template.xlsx')


logger.debug('MAIN ENDS')
# Note: the logger can be used only *after* ragtime.config.init_project
logger.debug(f'{PROJECT_NAME} STARTS')