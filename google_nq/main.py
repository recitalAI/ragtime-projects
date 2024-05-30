PROJECT_NAME:str = "google_nq"

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
# expe:Expe = Expe(json_path=FOLDER_ANSWERS / "google_nq.json")
# new_system_for_facts:str = """Génère un minimum de phrases numérotées courtes et simples qui décrivent ce paragraphe.
# Chaque phrase ne doit contenir qu'une seule information.
# Chaque phrase doit être indépendante et aucune phrase ne doit contenir la même information qu'une autre phrase.
# Les phrases ne doivent pas contenir de référence au document source ni à sa page.
# Les phrases doivent être compréhensibles seules et donc ne pas contenir de référence aux autres phrases ni nécessiter les autres phrases pour être comprises."""
# prompter:PptrFactsFR = PptrFactsFR(system=new_system_for_facts)
# facts_gen:FactGenerator = FactGenerator(llms="gpt-4o", prompter=prompter)
# facts_gen.generate(expe=expe)
# expe.save_to_json()
# expe.save_to_html()

##### Generate EVALS
expe:Expe = Expe(json_path=FOLDER_ANSWERS / "google_nq.json")
eval_gen:EvalGenerator = EvalGenerator(llms = "gpt-4o", prompter=EvalPrompterFR())
eval_gen.generate(expe=expe)
expe.save_to_json(path=FOLDER_EVALS)
expe.save_to_html(path=FOLDER_EVALS)

# rendering_params:dict = DEFAULT_HTML_RENDERING
# rendering_params['show_answers'] = False
# expe.save_to_html(render_params=rendering_params)

# ans_gen:AnsGenerator = AnsGenerator(retriever=None,
#                                     llm_names=["gpt-4", 'vertex_ai/gemini-pro', "mistral/mistral-large-latest",
#                                               "groq/llama3-8b-8192", "groq/llama3-70b-8192",
#                                               "groq/mixtral-8x7b-32768", "groq/gemma-7b-it"],
#                                     prompter=PptrAnsBase())
# expe:Expe = generators.generate(text_generator=ans_gen,
#                                 folder_in=FOLDER_FACTS,
#                                 folder_out=FOLDER_ANSWERS,
#                                 json_file="validation_set--30Q_0C_219F_0M_0A_0HE_0AE_2024-05-02_17h30,58.json",
#                                 save_to_html=True,
#                                 save_to_spreadsheet=True,
#                                 template_spreadsheet_path=FOLDER_SST_TEMPLATES / 'without_retriever.xlsx')

# eval_gen:EvalGenerator = EvalGenerator(llm_names=["gpt-4"], prompter=PptrEvalFR())
# expe = generators.generate(text_generator=eval_gen,
#                            folder_in=FOLDER_ANSWERS,
#                            folder_out=FOLDER_EVALS,
#                            # json_file=expe.json_path.stem + '.json',
#                            json_file='validation_set--30Q_0C_219F_7M_210A_0HE_0AE_2024-05-08_18h48,57.json',
#                            save_to_html=True,
#                            save_to_spreadsheet=True,
#                            template_spreadsheet_path=FOLDER_SST_TEMPLATES / 'without_retriever.xlsx')

logger.debug('MAIN ENDS')