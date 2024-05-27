PROJECT_NAME:str = "what_do_LLM_think"

import ragtime
from classes import ( MCQAnsPptr )
from ragtime.pipeline import (
    LLMs_from_names,
    run_pipeline,
)
from albert_llm_api_caller import ( Albert_LLM )

# always start with init_project before importing ragtime.config values since they are updated
# with init_project and import works by value and not by reference, so values imported before
# calling init_project are not updated after the function call
ragtime.config.init_project(
    name = PROJECT_NAME,
    init_type = "globals_only"
)
from ragtime.config import FOLDER_ANSWERS, FOLDER_QUESTIONS, logger, FOLDER_SST_TEMPLATES

llms_name:list[str] = [
    "gpt-4",
    "gpt-3.5-turbo",
    "mistral/mistral-large-latest",
    "groq/llama3-8b-8192",
    "groq/llama3-70b-8192",
    "groq/mixtral-8x7b-32768",
    "groq/gemma-7b-it",
]

prompter = MCQAnsPptr()
llms = LLMs_from_names(
    prompter = prompter,
    names = llms_name
)
llms.append(Albert_LLM(prompter = prompter))


configuration:dict = {
    'file_name': "economics--47Q_0C_0F_7M_329A_0HE_0AE_2024-03-14_19h17,45.json",
    'folder_name': FOLDER_ANSWERS,
    'generate': {
        'answers': {
            'retriever': None,
            'llms': llms,
            'export': {
                'spreadsheet': {
                    'path': FOLDER_SST_TEMPLATES / 'MCQ.xlsx',
                }
            },
        },
    }
}

run_pipeline(configuration = configuration)
