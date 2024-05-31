PROJECT_NAME: str = "Albert"

import ragtime
from ragtime.pipeline import run_pipeline, LLMs_from_names
from ragtime.prompters import FactPrompterFR, EvalPrompterFR
from classes import LLM, Albert_LLM, Prompter_from_human_evaluated_Expe

# always start with init_project before importing ragtime.config values since they are updated
# with init_project and import works by value and not by reference, so values imported before
# calling init_project are not updated after the function call
ragtime.config.init_project(name=PROJECT_NAME, init_type="globals_only")
from ragtime.config import (
    FOLDER_ANSWERS,
    FOLDER_FACTS,
    FOLDER_EVALS,
    FOLDER_SST_TEMPLATES,
)


def pipeline(start_form: str = None, stop_after: str = None):
    prompter = Prompter_from_human_evaluated_Expe()
    llms_for_answers_generator: list[LLM] = []

    llms_for_answers_generator.extend(
        LLMs_from_names(
            prompter=prompter,
            names=[
                "gpt-4o",
                "mistral/mistral-large-latest",
                "groq/llama3-8b-8192",
                "groq/llama3-70b-8192",
                "groq/mixtral-8x7b-32768",
                "groq/gemma-7b-it",
            ],
        )
    )
    llms_for_answers_generator.append(Albert_LLM(prompter=prompter))

    llms_for_facts_generator: list[LLM] = LLMs_from_names(
        prompter=FactPrompterFR(), names=["gpt-4o"]
    )

    llms_for_evals_generator: list[LLM] = LLMs_from_names(
        prompter=EvalPrompterFR(), names=["gpt-4o"]
    )

    configuration: dict = {
        "file_name": "HF_QCM_reconcilie_V1.json",
        "folder_name": FOLDER_ANSWERS,
        "generate": {
            "answers": {
                "llms": llms_for_answers_generator,
                "export": {"html": {}},
            },
            "facts": {
                "llms": llms_for_facts_generator,
                "export": {"html": {}},
            },
            "evals": {
                "llms": llms_for_evals_generator,
                "export": {"html": {}},
            },
        },
    }

    run_pipeline(
        start_from=start_form,
        stop_after=stop_after,
        configuration=configuration,
    )


from ragtime.expe import Expe

file_name: str = (
    "HF_QCM_reconcilie_V1--33Q_0C_431F_7M_231A_231HE_216AE_2024-05-31_15h54,52.json"
)
expe: Expe = Expe(FOLDER_EVALS / file_name)

expe.save_to_spreadsheet(
    path=FOLDER_EVALS / file_name,
    template_path=FOLDER_SST_TEMPLATES / "Test_template.xlsx",
)
