PROJECT_NAME:str = "google_nq"

import ragtime
from ragtime import expe, generators
from ragtime.expe import QA, Chunks, Prompt, Question, WithLLMAnswer
from ragtime.generators import StartFrom, PptrFactsFRv2, PptrEvalFRv2
from ragtime.expe import Expe

# always start with init_project before importing ragtime.config values since they are updated
# with init_project and import works by value and not by reference, so values imported before
# calling init_project are not updated after the function call
ragtime.config.init_project(name=PROJECT_NAME, init_type="globals_only")
from ragtime.config import FOLDER_ANSWERS, FOLDER_QUESTIONS, FOLDER_FACTS, FOLDER_EVALS, logger, FOLDER_SST_TEMPLATES
ragtime.config.init_win_env(['OPENAI_API_KEY', 'ALEPHALPHA_API_KEY', 'ANTHROPIC_API_KEY',
                             'COHERE_API_KEY', 'HUGGINGFACE_API_KEY', 'MISTRAL_API_KEY'])


logger.debug('MAIN STARTS')

# generators.gen_Facts(folder_in=FOLDER_ANSWERS, folder_out=FOLDER_FACTS, json_file='google_nq.json',
#                      llm_names=['gpt-4'], prompter=PptrFactsFRv2())

# generators.gen_Evals(folder_in=FOLDER_FACTS, folder_out=FOLDER_EVALS, 
#                      json_file='google_nq--30Q_0C_221F_1M_30A_30HE_0AE_2024-03-16_16h53,14.json',
#                      llm_names=['gpt-4'], prompter=PptrEvalFRv2())


expe.export_to_html(json_path=FOLDER_EVALS / "google_nq--30Q_0C_221F_1M_30A_30HE_30AE_2024-04-01_00h47,35.json")
expe.export_to_spreadsheet(json_path=FOLDER_EVALS / "google_nq--30Q_0C_221F_1M_30A_30HE_30AE_2024-04-01_00h47,35.json",
                           template_path=FOLDER_SST_TEMPLATES/'test_facts.xlsx')


logger.debug('MAIN ENDS')