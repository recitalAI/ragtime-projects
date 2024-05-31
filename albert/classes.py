from ragtime.base import call_api, REQ_POST
from ragtime.prompters import Prompter, Prompt
from ragtime.llms import LLM

from ragtime.expe import QA, Chunks, Prompt, Question, Answer, Eval, LLMAnswer


from typing import Optional
from datetime import datetime
import sseclient
import asyncio
import os

ALBERT_EMAIL: str = os.getenv("ALBERT_EMAIL")
ALBERT_USERNAME: str = os.getenv("ALBERT_USERNAME")
ALBERT_PASSWORD: str = os.getenv("ALBERT_PASSWORD")

ALBERT_URL = "https://albert.etalab.gouv.fr/api/v2"
ALBERT_SIGNIN: str = ALBERT_URL + "/sign_in"
ALBERT_STREAM: str = ALBERT_URL + "/stream"
ALBERT_FETCH_STREAM: str = ALBERT_URL + "/stream/{stream_id}/start"


class Albert_LLM(LLM):
    name: str = "AgentPublic/albertlight-7b"
    _model_name: str = "AgentPublic/albertlight-7b"
    _temperature: int = 0
    _token: str = ""
    _headers: dict = {}
    _token_last_update: datetime = datetime.now()
    _TOKEN_DURATION: int = 24  # max token duration in hours
    _num_retries: int = 3

    async def _refresh_token(self):
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
        response = await asyncio.to_thread(call_api, **request)
        json = response.json()
        self._token = json.get("token")
        self._token_last_update = datetime.now()

    async def _init_stream(self, query: str, with_history: bool = False):
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
        response = await asyncio.to_thread(call_api, **request)
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
                await self._refresh_token()
                stream_id = await self._init_stream(prompt.user)
                result = self.fetch_stream(stream_id=stream_id)
                break
            except:
                await asyncio.sleep(time_to_wait)
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


class Prompter_from_human_evaluated_Expe(Prompter):
    """
    This simple prompter just send the question as is to the LLM
    and does not perform any post-processing
    """

    system: str = ""

    def get_prompt(self, question: Question, chunks: Optional[Chunks] = None) -> Prompt:
        result: Prompt = Prompt()
        result.user = f"{question.text}"
        result.system = self.system
        return result

    def post_process(self, qa: QA = None, cur_obj: Answer = None):
        """
        Does not do anything by default, but can be overridden to add fields in meta data for instance
        """
        cur_obj.text = cur_obj.llm_answer.text
        if len(qa.answers.items) > 0:
            cur_obj.eval = Eval(
                text=qa.answers.items[0].text,
                human=qa.answers.items[0].eval.human,
            )
