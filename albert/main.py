PROJECT_NAME: str = "Albert"

import ragtime
from ragtime.base.llm import LLM
import ragtime.pipeline
from albert_llm_api_caller import Albert_LLM
from prompter_MCQ import MCQAnsPptr

# always start with init_project before importing ragtime.config values since they are updated
# with init_project and import works by value and not by reference, so values imported before
# calling init_project are not updated after the function call
ragtime.config.init_project(name=PROJECT_NAME, init_type="globals_only")
from ragtime.config import (
    FOLDER_ANSWERS,
    FOLDER_SST_TEMPLATES,
)


def LLMs():
    prompter = MCQAnsPptr()
    llms: list[LLM] = []
    # llms = ragtime.generators.LLMs_from_names(
    #     prompter=prompter,
    #     names=[
    #         "gpt-4",
    #         "gpt-3.5-turbo",
    #         "gpt-4",
    #         "mistral/mistral-large-latest",
    #         "groq/llama3-8b-8192",
    #         "groq/llama3-70b-8192",
    #         "groq/mixtral-8x7b-32768",
    #         "groq/gemma-7b-it",
    #     ],
    # )
    llms.append(Albert_LLM(prompter=prompter))
    return llms


configuration: dict = {
    "folder_name": FOLDER_ANSWERS,
    "file_name": "HF_QCM_reconcilie_V1.json",
    "generate": {
        "answers": {
            "llms": LLMs(),
            "b_overwrite": True,
            "export": {
                "json": {},
                "spreadsheet": {
                    "path": FOLDER_SST_TEMPLATES / "MCQ.xlsx",
                },
            },
        },
    },
}

ragtime.pipeline.run_pipeline(configuration=configuration)
