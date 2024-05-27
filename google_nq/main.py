PROJECT_NAME:str = "google_nq"

import ragtime
from ragtime.pipeline import (
    LLMs_from_names,
    run_pipeline,
)
import ragtime.prompters as prompters

# always start with init_project before importing ragtime.config values since they are updated
# with init_project and import works by value and not by reference, so values imported before
# calling init_project are not updated after the function call
ragtime.config.init_project(name=PROJECT_NAME, init_type="globals_only")
from ragtime.config import FOLDER_ANSWERS, FOLDER_SST_TEMPLATES
ragtime.config.init_win_env([
    'OPENAI_API_KEY',
    'ALEPHALPHA_API_KEY',
    'ANTHROPIC_API_KEY',
    'COHERE_API_KEY',
    'HUGGINGFACE_API_KEY',
    'MISTRAL_API_KEY',
    'NLP_CLOUD_API_KEY',
    'GROQ_API_KEY'
])

export:dict = {
    'html': dict(),
    'json': dict(),
    'spreadsheet': {
        'path': FOLDER_SST_TEMPLATES / 'without_retriever.xlsx'
    }
}

configuration:dict = {
    'file_name': "google_nq.json",
    'folder_name': FOLDER_ANSWERS,
    'generate': {
        'answers': {
            'llms': LLMs_from_names(
                prompter = prompters.table['PptrAnsBase'](),
                names = [
                    "gpt-4",
                    "gpt-3.5-turbo",
                    "mistral/mistral-large-latest",
                    "groq/llama3-8b-8192",
                    "groq/llama3-70b-8192",
                    "groq/mixtral-8x7b-32768",
                    "groq/gemma-7b-it",
                ]),
            'export': export,
        },
        'evals': {
            'llms': LLMs_from_names(
                prompter = prompters.table['PptrEvalFR'](),
                names = ["gpt-4"]),
            'export': export,
        }
    }
}

run_pipeline(
    configuration = configuration,
    start_from = 'evals'
    )
