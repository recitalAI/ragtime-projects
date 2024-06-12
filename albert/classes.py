from ragtime.base import call_api, REQ_POST, div0
from ragtime.prompters import Prompter, Prompt
from ragtime.llms import LLM
from ragtime.expe import QA, Prompt, Answer, Facts, Eval, LLMAnswer
from ragtime.config import logger


from datetime import datetime
import sseclient
import asyncio
import os
import re


ALBERT_EMAIL: str = os.getenv("ALBERT_EMAIL")
ALBERT_USERNAME: str = os.getenv("ALBERT_USERNAME")
ALBERT_PASSWORD: str = os.getenv("ALBERT_PASSWORD")

ALBERT_URL = "https://albert.etalab.gouv.fr/api/v2"
ALBERT_SIGNIN: str = ALBERT_URL + "/sign_in"
ALBERT_STREAM: str = ALBERT_URL + "/stream"
ALBERT_FETCH_STREAM: str = ALBERT_URL + "/stream/{stream_id}/start"

MODEL_NAME = "AgentPublic/llama3-instruct-8b"


class Albert_LLM(LLM):
    name: str = MODEL_NAME
    _model_name: str = MODEL_NAME
    _temperature: int = 0
    _token: str = ""
    _headers: dict = {}
    _token_last_update: datetime = datetime.now()
    _TOKEN_DURATION: int = 24  # max token duration in hours
    _num_retries: int = 3

    def _refresh_token(self):
        if self._token:
            # Calculate the difference in hours between now and the last token update
            diff_in_seconds = (datetime.now() - self._token_last_update).total_seconds()
            diff_in_hours = diff_in_seconds / 3600  # Convert seconds to hours
            if diff_in_hours < self._TOKEN_DURATION:
                return
        request = {
            "a_req_type": REQ_POST,
            "a_url": ALBERT_SIGNIN,
            "headers": {
                "Content-Type": "application/json",
            },
            "json": {
                "email": ALBERT_EMAIL,
                "username": ALBERT_USERNAME,
                "password": ALBERT_PASSWORD,
            },
        }
        # response = await asyncio.to_thread(call_api, **request)
        response = call_api(**request)
        json = response.json()
        self._token = json.get("token")
        self._token_last_update = datetime.now()

    def _init_stream(self, query: str, with_history: bool = False):
        request = {
            "a_req_type": REQ_POST,
            "a_url": ALBERT_STREAM,
            "headers": {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._token}",
            },
            "json": {
                "model_name": self._model_name,
                "mode": "rag",
                "query": query,
                "limit": 7,
                "with_history": with_history,
                #'context': ??
                #'institution': ??
                #'links': ??
                "temperature": self._temperature,
                "sources": ["service-public", "travail-emploi"],
                # "should_sids": [],
                # "must_not_sids": [],
                #'postprocessing': ??
            },
        }
        response = call_api(**request)
        json = response.json()
        stream_id = json.get("id")
        return stream_id

    def fetch_stream(self, stream_id: int) -> str:
        raw_data: str = ""
        # Create an SSE client
        messages = sseclient.SSEClient(
            url=ALBERT_FETCH_STREAM.format(stream_id=stream_id),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._token}",
                "Connection": "keep-alive",
            },
        )

        # Process the stream
        for msg in messages:
            tmp: str = msg.data.encode().decode("unicode_escape").strip('"')
            if tmp == "[DONE]":
                break
            raw_data = "".join([raw_data, tmp])
        return raw_data

    # TODO:
    # keep an eye on the rate limite
    async def complete(self, prompt: Prompt) -> LLMAnswer:
        retry: int = 1
        time_to_wait: float = 3.0
        result: str = ""
        while retry < self._num_retries:
            try:
                start_ts: datetime = datetime.now()
                self._refresh_token()
                stream_id = self._init_stream(prompt.user)
                result = self.fetch_stream(stream_id=stream_id)
                break
            except Exception as e:
                await asyncio.sleep(time_to_wait)
                logger.debug(
                    f"Excepton occured during api call, will retry 3 time each 3 second interval\nERROR:\n{e}"
                )
                retry += 1

        duration = (datetime.now() - start_ts).total_seconds()
        # is the duration of the api call
        cost = 0.0  # is the cost issued from a api call

        return LLMAnswer(
            name=self.name,
            prompt=prompt,
            text=result,  # need refacto see albert doc,
            full_name=self._model_name,  # full name of the model?
            timestamp=start_ts,
            duration=duration,
            cost=cost,
        )


class EvalPrompterLSA(Prompter):
    """
    Prompt: FAITS and REPONSE - expect the REPONSE to be rewritten including the FACTS in the text
    Post_process: analyse cited factsfacts not cited, and facts invented (?)
    """

    system: str = """Vous devez comparer une liste numérotée de FAITS avec une REPONSE. Votre tâche consiste à évaluer la présence de chaque FAIT dans la REPONSE, en suivant ces règles :

Reproduire la REPONSE telle quelle, en insérant entre parenthèses le numéro de chaque FAIT dont l'idée principale est présente dans la REPONSE.
Si une phrase de la REPONSE correspond à plusieurs FAITS, indiquer les numéros de ces FAITS entre parenthèses, séparés par une virgule (ex : (1,2)).
Ne valider un FAIT que si toute son idée principale est clairement exprimée dans la REPONSE ou peut être déduite de la REPONSE dans son ensemble.
Si une partie de la REPONSE ne correspond à aucun FAIT, insérer un point d'interrogation entre parenthèses (?) à cet endroit.
Si une partie de la REPONSE fait référence à un emplacement dans un document (ex : page X), ne rien indiquer pour cette partie.

Quelques précisions :

Ne vous fiez pas à la formulation exacte des FAITS, mais concentrez-vous sur leurs idées principales.
Une idée peut être exprimée différemment dans la REPONSE par rapport aux FAITS, l'essentiel est que le sens soit le même.
Si une idée est sous-entendue ou peut être déduite de l'ensemble de la REPONSE, vous pouvez valider le FAIT correspondant.
        """

    def get_prompt(self, answer: Answer, facts: Facts) -> Prompt:
        result: Prompt = Prompt()
        facts_as_str: str = "\n".join(
            f"{i}. {fact.text}" for i, fact in enumerate(facts, start=1)
        )
        result.user = f"-- FAITS --\n{facts_as_str}\n\n-- REPONSE --\n{answer.text}"
        result.system = self.system
        return result

    def post_process(self, qa: QA, cur_obj: Eval):
        answer: str = cur_obj.llm_answer.text if cur_obj.llm_answer.text != "[]" else ""
        # removes the word FAIT before the fact number as it is sometimes generated in the answer
        answer = answer.replace("(FAIT ", "(")
        # get the set of facts numbers from answer
        facts_in_answer: set[int] = set(
            [
                int(s)
                for s in ",".join(re.findall("\([\d+,+\s+]+\)", answer))
                .replace("(", "")
                .replace(")", "")
                .split(",")
                if s
            ]
        )
        # get the numbers in the true facts
        true_facts: set[int] = set(
            [int(s.text[0] if s.text[1] == "." else s.text[:2]) for s in qa.facts if s]
        )
        true_facts_in_answer: set[int] = facts_in_answer & true_facts
        true_facts_not_in_answer: set[int] = true_facts - true_facts_in_answer
        # get the number of extra facts (?) - they are not always hallucinations, sometimes just true facts not interesting and not included as usefule facts
        nb_extra_facts_in_answer: int = len(re.findall("\(\?\)", answer))
        # compute metrics
        precision: float = div0(
            len(true_facts_in_answer), len(facts_in_answer) + nb_extra_facts_in_answer
        )
        recall: float = div0(len(true_facts_in_answer), len(true_facts))
        cur_obj.meta["precision"] = precision
        cur_obj.meta["recall"] = recall
        cur_obj.meta["extra"] = nb_extra_facts_in_answer
        cur_obj.meta["missing"] = ", ".join(
            str(v) for v in list(true_facts_not_in_answer)
        )
        cur_obj.meta["nb_missing"] = len(true_facts_not_in_answer)
        cur_obj.meta["facts_in_ans"] = str(sorted(facts_in_answer))
        cur_obj.auto = div0(2 * precision * recall, precision + recall)
        cur_obj.text = answer
