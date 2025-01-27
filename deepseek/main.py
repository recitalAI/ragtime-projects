PROJECT_NAME:str = "deepseek"

import ragtime
from ragtime import expe, generators
from ragtime.expe import QA, Chunks, Prompt, Expe, Answer, Question, WithLLMAnswer
import json
from pathlib import Path

# always start with init_project before importing ragtime.config values since they are updated
# with init_project and import works by value and not by reference, so values imported before
# calling init_project are not updated after the function call
ragtime.config.init_project(name=PROJECT_NAME, init_type="globals_only")
from ragtime.config import FOLDER_ANSWERS, FOLDER_QUESTIONS, logger

# Note: the logger can be used only *after* ragtime.config.init_project
logger.debug(f'{PROJECT_NAME} STARTS')

# If you're using Windows, make your environment variables for LLM providers accessible with the following instruction
# ragtime.config.init_win_env(['OPENAI_API_KEY', 'ALEPHALPHA_API_KEY', 'MISTRAL_API_KEY'])

expe:Expe = Expe()

with open(FOLDER_QUESTIONS / "Culture_Validation_set_100Q.json") as f:
    json_all_qa:dict = json.load(f)

for json_qa in json_all_qa:
    qa:QA = QA()
    qa.question.text = json_qa["question"]
    qa.answers.append(Answer(text=json_qa["answer"]))
    expe.append(qa)

expe.save_to_json(FOLDER_ANSWERS / "cultural_qa")