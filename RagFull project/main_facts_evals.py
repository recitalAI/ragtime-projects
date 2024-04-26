PROJECT_NAME:str = "RagFull project"


import ragtime
from ragtime import expe, generators
from ragtime.generators import PptrFactsFRv2, PptrEvalFRv2
from RAG import load_env


# always start with init_project before importing ragtime.config values since they are updated
# with init_project and import works by value and not by reference, so values imported before
# calling init_project are not updated after the function call
ragtime.config.init_project(name=PROJECT_NAME, init_type="globals_only")
from ragtime.config import FOLDER_ANSWERS, FOLDER_QUESTIONS, logger, FOLDER_SST_TEMPLATES, FOLDER_FACTS, FOLDER_EVALS
ragtime.config.init_win_env(['OPENAI_API_KEY'])

load_env()

logger.debug('MAIN STARTS')

# generators.gen_Facts(folder_in=FOLDER_ANSWERS, folder_out=FOLDER_FACTS, json_file='questions--16Q_64C_0F_1M_16A_0HE_0AE_2024-04-26_01h55,36.json',
#                      llm_names=['gpt-3.5-turbo'], prompter=PptrFactsFRv2())

# generators.gen_Evals(folder_in=FOLDER_FACTS, folder_out=FOLDER_EVALS, 
#                      json_file='questions--16Q_64C_81F_1M_16A_16HE_0AE_2024-04-26_02h16,44.json',
#                      llm_names=['gpt-3.5-turbo'], prompter=PptrEvalFRv2())

expe.export_to_html(json_path=FOLDER_EVALS / "questions--16Q_64C_81F_1M_16A_16HE_12AE_2024-04-26_02h18,45.json")
expe.export_to_spreadsheet(json_path=FOLDER_EVALS / "questions--16Q_64C_81F_1M_16A_16HE_12AE_2024-04-26_02h18,45.json",
                           template_path=FOLDER_SST_TEMPLATES/'spreadsheet_rich_template.xlsx')


logger.debug('MAIN ENDS')

