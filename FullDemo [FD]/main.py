from ragtime.config import (
    FOLDER_ANSWERS,
    FOLDER_FACTS,
    FOLDER_EVALS,
    FOLDER_HTML_TEMPLATES,
    FOLDER_SST_TEMPLATES
)
from ragtime.expe import Expe, QA
from pathlib import Path
from ragtime.generators import QuestAnsGenerator, FactGenerator, EvalGenerator
from classes import EvalPrompterAlbert
from ragtime.llms import LLM
from ragtime.prompters import (
    Prompter,
    QuestAnsPrompterFR,
    FactPrompterJazz,
    EvalPrompterFR,
)
from ragtime.retrievers import annotation_human_auto
from ragtime.pipeline import LLMs_from_names
import ragtime
PROJECT_NAME: str = "FullDemo[FD]"


# always start with init_project before importing ragtime.config values since they are updated
# with init_project and import works by value and not by reference, so values imported before
# calling init_project are not updated after the function call
ragtime.config.init_project(name=PROJECT_NAME, init_type="globals_only")

#############################
#   Generators
#############################


def questions_answers_generator(nb_quest: int, docs_path: Path) -> Path:
    expe: Expe = Expe()

    prompter: Prompter = QuestAnsPrompterFR()

    llms_for_answers_generator: list[LLM] = LLMs_from_names(
        prompter=prompter, names=["gpt-4o"]
    )

    answer_generator: QuestAnsGenerator = QuestAnsGenerator(
        nb_quest=nb_quest, docs_path=docs_path, llms=llms_for_answers_generator)
    answer_generator.generate(expe=expe)

    output_path: Path = FOLDER_ANSWERS / docs_path
    path_to_return: Path = expe.save_to_json(path=output_path)
    expe.save_to_html(
        path=output_path,
        template_path=FOLDER_HTML_TEMPLATES / "basic_template.jinja",
    )
    return path_to_return


def facts_generator(path: Path) -> Path:
    expe: Expe = Expe(path)

    prompter: Prompter = FactPrompterJazz()

    eval_gen: FactGenerator = FactGenerator(llms=["gpt-4o"], prompter=prompter)
    eval_gen.generate(expe=expe)

    output_path: Path = FOLDER_FACTS / path.name
    path_to_return: Path = expe.save_to_json(path=output_path)
    expe.save_to_html(
        path=output_path,
        template_path=FOLDER_HTML_TEMPLATES / "basic_template.jinja",
    )
    return path_to_return


def evals_generator(path: Path, prompter: Prompter) -> Path:
    expe: Expe = Expe(path)

    eval_gen: EvalGenerator = EvalGenerator(llms="gpt-4", prompter=prompter)
    eval_gen.generate(expe=expe)

    output_path: Path = FOLDER_EVALS / path.name
    path_to_return: Path = expe.save_to_json(path=output_path)
    expe.save_to_html(
        path=output_path,
        template_path=FOLDER_HTML_TEMPLATES / "basic_template.jinja",
    )
    expe.save_to_spreadsheet(
        path=output_path,
        template_path=FOLDER_SST_TEMPLATES / "Eval_template.xlsx",
    )
    return path_to_return


if __name__ == "__main__":
    nb_quest = 10
    docs_path = 'Test'
    file_path = questions_answers_generator(
        nb_quest=nb_quest, docs_path=docs_path)
    expe = annotation_human_auto(path=FOLDER_ANSWERS /
                                 file_path)
    file_path = facts_generator(path=expe.json_path)
    evals_generator(path=file_path, prompter=EvalPrompterAlbert())
