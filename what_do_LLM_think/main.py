PROJECT_NAME:str = "what_do_LLM_think"

import ragtime
from ragtime import expe, generators
from ragtime.expe import QA, Chunks, Prompt, Question, WithLLMAnswer
import keys
from classes import MCQAnsPptr
from ragtime.generators import StartFrom
from ragtime.expe import Expe

# always start with init_project before importing ragtime.config values since they are updated
# with init_project and import works by value and not by reference, so values imported before
# calling init_project are not updated after the function call
ragtime.config.init_project(name=PROJECT_NAME, init_type="globals_only")
from ragtime.config import FOLDER_ANSWERS, FOLDER_QUESTIONS, logger, FOLDER_SST_TEMPLATES

logger.debug('MAIN STARTS')

# generators.gen_Answers(folder_in=FOLDER_ANSWERS, folder_out=FOLDER_ANSWERS,
#                         json_file='economics--47Q_0C_0F_7M_329A_0HE_0AE_2024-03-14_17h58,57.json',
#                         prompter=MCQAnsPptr(), b_missing_only=True,
#                         llm_names=["gpt-4", "gemini-pro", "command-nightly", "gpt-3.5-turbo", "mistral/mistral-large-latest", 
#                                    "claude-3-opus-20240229", "luminous-supreme-control"])

# expe.export_to_html(json_path=FOLDER_ANSWERS / filename)
expe.export_to_spreadsheet(json_path=FOLDER_ANSWERS / "economics--47Q_0C_0F_7M_329A_0HE_0AE_2024-03-14_19h17,45.json",
                           template_path=FOLDER_SST_TEMPLATES / 'MCQ.xlsx')


logger.debug('MAIN ENDS')