PROJECT_NAME: str = "albert"

import ragtime
from ragtime import expe, generators
from ragtime.expe import QA, Chunks, Prompt, Question, WithLLMAnswer, Expe, UpdateTypes
from ragtime.generators import FactGenerator, EvalGenerator
from ragtime.prompters import FactPrompterJazz, EvalPrompterFR
import logging
# import litellm
# litellm.set_verbose=True

# always start with init_project before importing ragtime.config values since they are updated
# with init_project and import works by value and not by reference, so values imported before
# calling init_project are not updated after the function call
ragtime.config.init_project(name=PROJECT_NAME, init_type="globals_only")
from ragtime.config import FOLDER_ANSWERS, FOLDER_FACTS, FOLDER_EVALS, FOLDER_QUESTIONS, logger, FOLDER_SST_TEMPLATES, FOLDER_HTML_TEMPLATES
ragtime.config.init_win_env(['OPENAI_API_KEY', 'ALEPHALPHA_API_KEY', 'COHERE_API_KEY', 'TOGETHERAI_API_KEY',
                             'HUGGINGFACE_API_KEY', 'MISTRAL_API_KEY', 'VERTEXAI_LOCATION', 'VERTEXAI_PROJECT'])

logger.debug('MAIN STARTS')
expe:Expe = Expe(json_path=FOLDER_EVALS / "extract_log_good_cleaned--160Q_0C_832F_1M_160A_160HE_159AE_2024-06-28_21h11,47.json")
expe.save_to_spreadsheet(template_path=FOLDER_SST_TEMPLATES / "eval_template.xlsx")
logger.debug('MAIN ENDS')

# #############################
# #   Generators
# #############################


# def answers_generator(path: Path) -> Path:
#     expe: Expe = Expe(path)

#     # Génération des réponses
#     # Prompter -> AnsPrompterBase
#     # LLMs -> groq/llama3-70b-8192, Albert_LLM

#     prompter: Prompter = AnsPrompterBase()

#     llms_for_answers_generator: list[LLM] = LLMs_from_names(
#         prompter=prompter, names=["groq/llama3-70b-8192"]
#     )
#     llms_for_answers_generator.append(Albert_LLM(prompter=prompter))

#     answer_generator: AnsGenerator = AnsGenerator(llms=llms_for_answers_generator)
#     answer_generator.generate(expe=expe)

#     output_path: Path = FOLDER_ANSWERS / path.name
#     path_to_return: Path = expe.save_to_json(path=output_path)
#     expe.save_to_html(
#         path=output_path,
#         template_path=FOLDER_HTML_TEMPLATES / "basic_template.jinja",
#         b_show_answers=False,
#     )
#     return path_to_return


# def facts_generator(path: Path) -> Path:
#     expe: Expe = Expe(path)

#     # Génération des faits
#     # Prompter -> FactPrompterJazz
#     # Faits générés par gpt-4

#     prompter: Prompter = FactPrompterJazz()

#     eval_gen: FactGenerator = FactGenerator(llms=["gpt-4"], prompter=prompter)
#     eval_gen.generate(expe=expe)

#     output_path: Path = FOLDER_FACTS / path.name
#     path_to_return: Path = expe.save_to_json(path=output_path)
#     expe.save_to_html(
#         path=output_path,
#         template_path=FOLDER_HTML_TEMPLATES / "basic_template.jinja",
#         b_show_answers=False,
#     )
#     return path_to_return


# def evals_generator(path: Path, prompter: Prompter) -> Path:
#     expe: Expe = Expe(path)

#     # Génération des évaluations
#     # Prompter -> EvalPrompterLSA
#     # Evaluations générés par gpt-4

#     eval_gen: EvalGenerator = EvalGenerator(llms="gpt-4", prompter=prompter)
#     eval_gen.generate(expe=expe)

#     output_path: Path = FOLDER_EVALS / path.name
#     path_to_return: Path = expe.save_to_json(path=output_path)
#     expe.save_to_html(
#         path=output_path,
#         template_path=FOLDER_HTML_TEMPLATES / "basic_template.jinja",
#         b_show_answers=False,
#     )
#     return path_to_return


# #############################
# #   Scenario
# #############################


# def scenario_classic(path: Path) -> Path:
#     """
#     Ceci est un scenario classic allant de la génération des faits d'après les réponses humaines jusqu'à la génération des évaluations
#     """
#     path = facts_generator(path)
#     path = answers_generator(path)
#     path = evals_generator(path, EvalPrompterFR())
#     return path


# def scenario_clean_bad_questions(path: Path) -> Path:
#     """
#     La liste suivante sont les questions supprimer du dataset d'origine car les réponses fournis ne permettent pas de générer des faits utilisable pour l'évaluation

#     "question": "Quel formulaire cerfa utiliser pour renouveler une Carte Nationale d’Identité (CNI) ?"
#     "answers": "Aucun des formulaires cerfa mentionnés ci-dessus."

#     "question": "Comment acheter un timbre fiscal en ligne pour payer les frais de renouvellement de la carte d'identité ?"
#     "answers": "Aucune des réponses n’est la bonne"

#     "question": "Comment renouveler son passeport en ligne et éviter d'avoir à utiliser un formulaire cartonné au guichet ?"
#     "answers": "Aucune des réponses n’est la bonne"

#     "question": "Quelle est la date limite pour déposer ma demande de départ à la retraite en tant que fonctionnaire ?"
#     "answers": "L'assurance Retraite traite uniquement  les carrières qui appartiennent au régime général "
#     """

#     questions_to_remove: list[str] = [
#         "Quel formulaire cerfa utiliser pour renouveler une Carte Nationale d’Identité (CNI) ?",
#         "Comment acheter un timbre fiscal en ligne pour payer les frais de renouvellement de la carte d'identité ?",
#         "Comment renouveler son passeport en ligne et éviter d'avoir à utiliser un formulaire cartonné au guichet ?",
#         "Quelle est la date limite pour déposer ma demande de départ à la retraite en tant que fonctionnaire ?",
#     ]

#     expe: Expe = Expe(path)
#     new_expe: Expe = Expe()
#     for qa in expe:
#         if qa.question.text not in questions_to_remove:
#             new_expe.append(qa)
#     return new_expe.save_to_json(path=path)


# def scenario_based_on_validation_sets() -> Path:
#     """
#     Ce scénario génére deux evaluation basé sur les réponses des llms évaluer.
#     """
#     path: Path = (
#         FOLDER_VALIDATION_SETS
#         / "2024-06-11_validation_set_GPT4o-EvalPrompterFR_29Q_86F.json"
#     )
#     path = answers_generator(path)
#     evals_generator(path, EvalPrompterFR())
#     evals_generator(path, EvalPrompterLSA())


# if __name__ == "__main__":
#     # Definissez un nom de fichier par le quel vous souhaitez débuter
#     # et le dossier dans le quel il se trouve
#     file_name: str = "HF_QCM_reconcilie_V1--33Q_0C_0F_0M_33A_33HE_0AE.json"
#     folder: Path = Path(
#         FOLDER_ANSWERS,
#         # FOLDER_FACTS,
#         # FOLDER_EVALS,
#         # FOLDER_VALIDATION_SETS,
#     )

#     # Choisissez parmis un des scenarios suivant.

#     # scenario_classic(folder / file_name)
#     # scenario_clean_bad_questions(folder / file_name)
#     # scenario_based_on_validation_sets()
