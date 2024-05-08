PROJECT_NAME:str = "google_nq"

import ragtime
from ragtime import expe, generators
from ragtime.expe import QA, Chunks, Prompt, Question, WithLLMAnswer, UpdateTypes, Answers
from ragtime.generators import StartFrom, PptrFactsFRv2, PptrEvalFRv2, PptrRichAnsFR, PptrBaseAns
from ragtime.expe import Expe

# always start with init_project before importing ragtime.config values since they are updated
# with init_project and import works by value and not by reference, so values imported before
# calling init_project are not updated after the function call
ragtime.config.init_project(name=PROJECT_NAME, init_type="globals_only")
from ragtime.config import FOLDER_ANSWERS, FOLDER_QUESTIONS, FOLDER_FACTS, FOLDER_EVALS, logger, FOLDER_SST_TEMPLATES
ragtime.config.init_win_env(['OPENAI_API_KEY', 'ALEPHALPHA_API_KEY', 'ANTHROPIC_API_KEY',
                             'COHERE_API_KEY', 'HUGGINGFACE_API_KEY', 'MISTRAL_API_KEY',
                             'NLP_CLOUD_API_KEY', 'GROQ_API_KEY'])


logger.debug('MAIN STARTS')

expe:Expe = generators.gen_Answers(folder_in=FOLDER_FACTS,
                                   folder_out=FOLDER_ANSWERS,
                                   json_file="validation_set--30Q_0C_219F_0M_0A_0HE_0AE_2024-05-02_17h30,58.json",
                                   prompter=PptrBaseAns(),
                                   llm_names=["gpt-4", 'vertex_ai/gemini-pro', "mistral/mistral-large-latest",
                                              "groq/llama3-8b-8192", "groq/llama3-70b-8192",
                                              "groq/mixtral-8x7b-32768", "groq/gemma-7b-it"])
expe.save_to_json()
expe.export_to_html()
expe.export_to_spreadsheet(template_path=FOLDER_SST_TEMPLATES / 'rich_ans_template.xlsx')


# generators.gen_Facts(folder_in=FOLDER_ANSWERS, folder_out=FOLDER_FACTS, json_file='google_nq.json',
#                      llm_names=['gpt-4'], prompter=PptrFactsFRv2())

# generators.gen_Evals(folder_in=FOLDER_FACTS, folder_out=FOLDER_EVALS, 
#                      json_file='google_nq--30Q_0C_221F_1M_30A_30HE_0AE_2024-03-16_16h53,14.json',
#                      llm_names=['gpt-4'], prompter=PptrEvalFRv2())


file_name:str = "validation_set--30Q_0C_219F_7M_210A_0HE_0AE_2024-05-02_23h27,38.json"
expe.export_to_html(json_path=FOLDER_ANSWERS / file_name)
expe.export_to_spreadsheet(json_path=FOLDER_ANSWERS / file_name,
                           template_path=FOLDER_SST_TEMPLATES / 'rich_ans_template.xlsx')


logger.debug('MAIN ENDS')