PROJECT_NAME:str = "google_nq"

import ragtime
from ragtime import expe, generators
from ragtime.expe import QA, Chunks, Prompt, Question, WithLLMAnswer, UpdateTypes, Answers
from ragtime.generators import (
    StartFrom,
    LLMs_from_names, LLM,
    AnsGenerator, EvalGenerator,
    run_pipeline,
)
from ragtime.prompters import (
    prompt_table,
    PptrFactsFR, PptrEvalFR, PptrAnsBase,
)
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
llms_name:list[str] = [
    "gpt-4",
    "vertex_ai/gemini-pro",
    "mistral/mistral-large-latest",
    "groq/llama3-8b-8192",
    "groq/llama3-70b-8192",
    "groq/mixtral-8x7b-32768",
    "groq/gemma-7b-it"
]

export:dict = {
    'html': dict(),
    'spreadsheet': {
        'path': FOLDER_SST_TEMPLATES / 'without_retriever.xlsx'
    }
}

configuration:dict = {
    'retriever': None,
    'folder': FOLDER_FACTS,
    'file_name': "validation_set--30Q_0C_219F_0M_0A_0HE_0AE_2024-05-02_17h30,58.json",
    'generate': {
        'answers': {
            'llms': LLMs_from_names(llms_name, PptrAnsBase()),
            'folder': FOLDER_ANSWERS,
            'export': export,
        },
        'evals': {
            'llms': LLMs_from_names(["gpt-4"], PptrEvalFR()),
            'folder': FOLDER_EVALS,
            'export': export,
        }
    }
}

run_pipeline(configuration = configuration)

logger.debug('MAIN ENDS')



#def old_version():
#    llms_name:list[str] = [
#        "gpt-4",
#        "vertex_ai/gemini-pro",
#        "mistral/mistral-large-latest",
#        "groq/llama3-8b-8192",
#        "groq/llama3-70b-8192",
#        "groq/mixtral-8x7b-32768",
#        "groq/gemma-7b-it"
#    ]
#    llms:list[LLM] = LLMs_from_names(
#        names = llms_name,
#        prompter = prompt_table['PptrAnsBase']()
#    )
#
#    ans_gen:AnsGenerator = AnsGenerator(
#        retriever = None,
#        llms = llms,
#    )
#
#    expe:Expe = generators.generate(
#        text_generator = ans_gen,
#        folder_in = FOLDER_FACTS,
#        folder_out = FOLDER_ANSWERS,
#        json_file = "validation_set--30Q_0C_219F_0M_0A_0HE_0AE_2024-05-02_17h30,58.json",
#        save_to_html = True,
#        save_to_spreadsheet = True,
#        template_spreadsheet_path = FOLDER_SST_TEMPLATES / 'without_retriever.xlsx'
#    )
#
#    eval_gen:EvalGenerator = EvalGenerator(
#        llms = LLMs_from_names(
#            names = ["gpt-4"],
#            prompter = prompt_table['PptrEvalFR']()
#        ),
#    )
#
#    expe = generators.generate(
#        text_generator = eval_gen,
#        folder_in = FOLDER_ANSWERS,
#        folder_out = FOLDER_EVALS,
#        json_file = expe.json_path.stem + '.json',
#        #    json_file='validation_set--30Q_0C_219F_7M_209A_0HE_0AE_2024-05-11_19h51,17.json',
#        save_to_html = True,
#        save_to_spreadsheet = True,
#        template_spreadsheet_path = FOLDER_SST_TEMPLATES / 'without_retriever.xlsx'
#    )
