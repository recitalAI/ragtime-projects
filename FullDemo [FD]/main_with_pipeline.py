from ragtime.config import FOLDER_QUESTIONS, FOLDER_ANSWERS, FOLDER_FACTS, FOLDER_EVALS, logger, FOLDER_SST_TEMPLATES, FOLDER_HTML_TEMPLATES
from pathlib import Path
import litellm
import os
from ragtime.retrievers import annotation_human_auto
from ragtime.prompters import QuestAnsPrompterFR, AnsPrompterWithRetrieverFR, EvalPrompterFR, FactPrompterFR
from classes import EvalPrompterAlbert
from ragtime import pipeline
import ragtime

PROJECT_NAME: str = "Full Demo [FD]"


litellm.set_verbose = True


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

PATH = 'Test'

# always start with init_project before importing ragtime.config values since they are updated
# with init_project and import works by value and not by reference, so values imported before
# calling init_project are not updated after the function call
ragtime.config.init_project(name=PROJECT_NAME, init_type="globals_only")


logger.debug('MAIN STARTS')

export: dict = {
    'json': {},
    'html': {
        'path': FOLDER_HTML_TEMPLATES / 'LSA_template.jinja',
    },



}

configuration: dict = {
    'folder_name': FOLDER_FACTS,
    'file_name': '2024-04-03_107_questions--10Q_0C_22F_1M_10A_10HE_0AE_2024-06-25_17h34,43.json',
    'generate': {
        'questions': {
            'num_quest': 10,
            'docs_path': PATH,
            'llms': pipeline.LLMs_from_names(
                prompter=QuestAnsPrompterFR(),
                names=["gpt-4o"]
            ),
            'export': export,
        },
        'answers': {
            'docs_path': PATH,
            'llms': pipeline.LLMs_from_names(
                prompter=AnsPrompterWithRetrieverFR(),
                names=[
                    "gpt-4o",
                ]
            ),
            'export': export,
        },
        'facts': {
            'llms': pipeline.LLMs_from_names(
                prompter=FactPrompterFR(),
                names=["gpt-4"]
            ),
            'export': export,
        },
        'evals': {
            'llms': pipeline.LLMs_from_names(
                prompter=EvalPrompterAlbert(),
                # ou ['mistral/mistral-large-latest']
                names=["gpt-4"]
            ),
            'export': export,
        }
    }
}


# json_link = pipeline.run_pipeline(
#     configuration=configuration,
#     start_from='questions',
#     stop_after='questions',
# )

# expe = annotation_human_auto(path=FOLDER_QUESTIONS /
#                              Path(json_link))

# configuration['file_name'] = expe.json_path.stem + '.json'

pipeline.run_pipeline(
    configuration=configuration,
    start_from='evals',
    stop_after='evals',
)
