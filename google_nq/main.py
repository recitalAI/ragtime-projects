PROJECT_NAME: str = "google_nq"

import ragtime
from ragtime.expe import Expe
from ragtime.prompters import Prompter, FactPrompterFR, EvalPrompterFR
from ragtime.generators import FactGenerator, EvalGenerator

# always start with init_project before importing ragtime.config values since they are updated
# with init_project and import works by value and not by reference, so values imported before
# calling init_project are not updated after the function call
ragtime.config.init_project(name=PROJECT_NAME, init_type="globals_only")
from ragtime.config import FOLDER_ANSWERS, FOLDER_QUESTIONS, FOLDER_FACTS, FOLDER_EVALS, logger, FOLDER_SST_TEMPLATES, DEFAULT_HTML_RENDERING
ragtime.config.init_win_env(['OPENAI_API_KEY', 'ALEPHALPHA_API_KEY', 'ANTHROPIC_API_KEY',
                             'COHERE_API_KEY', 'HUGGINGFACE_API_KEY', 'MISTRAL_API_KEY',
                             'NLP_CLOUD_API_KEY', 'GROQ_API_KEY'])


logger.debug('MAIN STARTS')

##### Generate FACTS
expe:Expe = Expe(json_path=FOLDER_ANSWERS / "google_nq.json")
facts_gen:FactGenerator = FactGenerator(llms="gpt-4", prompter=FactPrompterFR())
facts_gen.generate(expe=expe)
expe.save_to_json(path=FOLDER_FACTS)
expe.save_to_html(path=FOLDER_FACTS)
expe.save_to_spreadsheet(path=FOLDER_FACTS)

# ##### Generate EVALS
# expe:Expe = Expe(json_path=FOLDER_FACTS / "google_nq--30Q_0C_307F_1M_30A_30HE_3AE_2024-06-02_09h14,22.json")
# eval_gen:EvalGenerator = EvalGenerator(llms = "gpt-4", prompter=EvalPrompterFR())
# eval_gen.generate(expe=expe)
# expe.save_to_json(path=FOLDER_EVALS)
# expe.save_to_html(path=FOLDER_EVALS, b_show_answers=False)
# expe.save_to_spreadsheet(path=FOLDER_EVALS)

logger.debug('MAIN ENDS')