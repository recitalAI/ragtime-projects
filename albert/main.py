PROJECT_NAME: str = "Albert"

import ragtime
from ragtime.pipeline import run_pipeline, LLMs_from_names
from ragtime.prompters import AnsPrompterBase, FactPrompterFR, EvalPrompterFR
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
    FOLDER_HTML_TEMPLATES,
)

name: str = "HF_QCM_reconcilie_V1{}.json"


def pipeline(
    start_form: str = None, stop_after: str = None, folder_name=None, suffixe=None
):
    prompter = AnsPrompterBase()  # Prompter_from_human_evaluated_Expe()
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
        prompter=EvalPrompterFR(), names=["gpt-4"]
    )

    configuration: dict = {
        "file_name": name.format(suffixe or ""),
        "folder_name": folder_name or FOLDER_ANSWERS,
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


# pipeline(
#     start_form="evals",
#     folder_name=FOLDER_FACTS,
#     suffixe="--33Q_0C_169F_8M_264A_33HE_0AE_2024-06-03_14h33,24",
# )


def merge_humanEvaluation_toGeneratedAnswers():
    from ragtime.expe import Expe, QA

    expe_human: Expe = Expe(FOLDER_ANSWERS / name.format(""))

    suffixe: str = "--33Q_0C_0F_7M_231A_0HE_0AE_2024-06-03_13h47,46"
    expe_to_complet: Expe = Expe(FOLDER_ANSWERS / name.format(suffixe))

    def haveSameQuestion(qa_a: QA, qa_b: QA):
        return qa_a.question.text == qa_b.question.text

    for qa_to_complet in expe_to_complet:
        # find the corresponding question in the human evaluated set
        human_qa = next(
            (item for item in expe_human if haveSameQuestion(item, qa_to_complet)),
            None,
        )
        if human_qa and len(human_qa.answers) > 0:
            qa_to_complet.answers.append(human_qa.answers[0])

    expe_to_complet.save_to_json(path=FOLDER_ANSWERS / name.format(""))


# from ragtime.expe import Expe

# file_name: str = (
#     "HF_QCM_reconcilie_V1--33Q_0C_431F_7M_231A_231HE_216AE_2024-05-31_15h54,52.json"
# )
# expe: Expe = Expe(FOLDER_EVALS / file_name)

# expe.save_to_spreadsheet(
#     path=FOLDER_EVALS / file_name,
#     template_path=FOLDER_SST_TEMPLATES / "Test_template.xlsx",
# )


from ragtime.expe import Expe


file_name: str = name.format("--33Q_0C_169F_8M_264A_33HE_200AE_2024-06-03_15h22,05")
expe: Expe = Expe(FOLDER_EVALS / file_name)

expe.save_to_html(
    path=FOLDER_EVALS / file_name,
    template_path=FOLDER_HTML_TEMPLATES / "basic_template.jinja",
    render_params={
        "show_answers": False,
        "show_chunks": True,
        "show_facts": True,
        "show_evals": True,
    },
)

# try to replay the post processing of a specific question / llm answer
#   - load the Expe
#       - find the question by similarity
#           - find the llm by name
#   - feed the Eval object to the corresponding post-processing
