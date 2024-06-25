
from ragtime.expe import QA, Prompt, Question, Answer, Eval, Facts
from ragtime.prompters import Prompter
import re
from ragtime.base import div0

class EvalPrompterAlbert(Prompter):
    """
    Prompt: FAITS and REPONSE - expect the REPONSE to be rewritten including the FACTS in the text
    Post_process: analyse cited factsfacts not cited, and facts invented (?)
    """

    system: str = """
    For each fact in a list of FACTS, determine whether the fact is supported in the PARAGRAPH or not and return :
- [OK] if the fact is supported, [NOT FOUND] if it is not supported and [HALLU] if an opposite fact is supported
- the reason why you return OK, NOT FOUND or HALLU
- the part in the PARAGRAPH related to the reason
At the end of the answer, add "[EXTRA] = number of ideas found in the PARAGRAPH that don't match the factual ideas." An idea is considered as [EXTRA] if:
-Off topic
-It gives information different from the facts ideas.
-Undesired extra context.
Exemple :
-> Input :

FACTS :
1. L'algorithme de Metropolis-Hastings cherche à obtenir une chaîne de Markov.
2. Cette chaîne de Markov doit admettre g comme mesure invariante.
3. Z ne doit pas apparaître dans le noyau de transition de cette chaîne.

PARAGRAPH
L'algorithme de Metropolis-Hastings cherche à obtenir une chaîne de Markov qui admette g comme mesure invariante et telle que Z n'apparaisse pas dans le noyau de transition.
-> Output :
1.[OK] - The paragraph states that "L'algorithme de Metropolis-Hastings cherche à obtenir une chaîne de Markov" which supports the first fact. 
Part in the paragraph: "L'algorithme de Metropolis-Hastings cherche à obtenir une chaîne de Markov"

2.[OK] - The paragraph states that "Cette chaîne de Markov doit admettre g comme mesure invariante" which supports the second fact.
Part in the paragraph: "qui admette g comme mesure invariante"

3.[OK] - The paragraph states that "Z ne doit pas apparaître dans le noyau de transition" which supports the third fact.
Part in the paragraph: "et telle que Z n'apparaisse pas dans le noyau de transition."

[EXTRA] = 0
        """

    def get_prompt(self, answer: Answer, facts: Facts) -> Prompt:
        result: Prompt = Prompt()
        facts_as_str: str = "\n".join(
            f"{i}. {fact.text}" for i, fact in enumerate(facts, start=1))
        result.user = f"-- FAITS --\n{facts_as_str}\n\n-- PARAGRAPH --\n{answer.text}"
        result.system = self.system
        return result

    def post_process(self, qa: QA, cur_obj: Eval):
        answer: str = cur_obj.llm_answer.text if cur_obj.llm_answer.text != "[]" else ""
        # removes the word FAIT before the fact number as it is sometimes generated in the answer
        answer = answer.replace("(FAIT ", "(")
        # get the set of facts numbers from answer
        facts_in_answer: set[int] = set([int(match) for match in re.findall(r'(\d+)\.\s*\[OK\]', answer)])
        hallus_in_answer: set[int] = set([int(match) for match in re.findall(r'(\d+)\.\s*\[HALLU\]', answer)])
        # get the numbers in the true facts
        true_facts: set[int] = set(
            [int(s.text[0] if s.text[1] == "." else s.text[:2]) for s in qa.facts if s])
        true_facts_in_answer: set[int] = facts_in_answer & true_facts
        hallus_in_answer: set[int] = hallus_in_answer & true_facts
        true_facts_not_in_answer: set[int] = true_facts - (true_facts_in_answer | hallus_in_answer)
        # get the number of extra facts (?) - they are not always hallucinations, sometimes just true facts not interesting and not included as usefule facts
        Extra = re.findall(r'\[EXTRA\]\s*=\s*(\d+)', answer)
        Extra_text = re.findall(r'\[EXTRA\]\s*=\s*\d+\s*(.*)', answer)
        nb_extra_facts_in_answer: int = int(Extra[0])

        # compute metrics
        cur_obj.meta["extra"] = " ".join(Extra_text)
        cur_obj.meta["nb_extra"] = nb_extra_facts_in_answer
        cur_obj.meta["missing"] = [i for i in true_facts_not_in_answer]
        cur_obj.meta["nb_missing"] = len(true_facts_not_in_answer)
        cur_obj.meta["ok"] = list(true_facts_in_answer)
        cur_obj.meta["nb_ok"] = len(true_facts_in_answer)
        cur_obj.meta["hallu"] = list(hallus_in_answer)
        cur_obj.meta["nb_hallu"] = len(hallus_in_answer)
        cur_obj.auto = max(0,div0(len(true_facts_in_answer) - len(hallus_in_answer),len(true_facts))-0.25*div0(len(true_facts_not_in_answer) + nb_extra_facts_in_answer,len(true_facts)))
        cur_obj.text = answer