#!/usr/bin/env python3

from datetime import datetime
from ragtime.base.data_type import Prompt, LLMAnswer
from ragtime.base.llm import LLM
import os
import asyncio
import sseclient
import requests

ALBERT_USERNAME:str = os.getenv("ALBERT_USERNAME")
ALBERT_EMAIL:str = os.getenv("ALBERT_EMAIL")
ALBERT_PASSWORD:str = os.getenv("ALBERT_PASSWORD")
ALBERT_URL_LOGIN:str = "https://albert.etalab.gouv.fr/api/v2/sign_in"

ALBERT_URL_CHAT_ID_STREAM:str = "https://albert.etalab.gouv.fr/api/v2/chat"
ALBERT_URL_INIT_STREAM:str = "https://albert.etalab.gouv.fr/api/v2/stream/chat/{chat_id}"
ALBERT_URL_FETCH_STREAM:str = "https://albert.etalab.gouv.fr/api/v2/stream/{stream_id}/start"

class Albert_LLM(LLM):
    name:str = "AgentPublic/albertlight-7b"
    _model_name:str = "AgentPublic/albertlight-7b"
    _temperature:int = 20
    _token:str = ""
    _headers:dict = {}
    _token_last_update: datetime = datetime.now()
    _TOKEN_DURATION:int = 24 # max token duration in hours

    async def _refresh_token(self):
        if self._token:
            # Calculate the difference in hours between now and the last token update
            diff_in_seconds = (datetime.now() - self._token_last_update).total_seconds()
            diff_in_hours = diff_in_seconds / 3600  # Convert seconds to hours
            if diff_in_hours < self._TOKEN_DURATION:
                return
        request = {
            'url': ALBERT_URL_LOGIN,
            'headers': {
                "Content-Type": "application/json",
            },
            'json': {
                "email": ALBERT_EMAIL,
                "password": ALBERT_PASSWORD,
            },
        }
        response = await asyncio.to_thread(requests.post, **request)
        json = response.json()
        self._token = json.get("token")
        self._token_last_update = datetime.now()

    async def get_chat_id(self):
        request = {
            'url': ALBERT_URL_CHAT_ID_STREAM,
            'headers': {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._token}"
            },
            'json': {
                "chat_type": "qa",
            },
        }
        response = await asyncio.to_thread(requests.post, **request)
        chat_id:int = response.json().get("id")
        return chat_id

    async def _init_stream(self, query:str, with_history:bool = False):
        chat_id = await self.get_chat_id()
        request = {
            'url': ALBERT_URL_INIT_STREAM.format(chat_id = chat_id),
            'headers': {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._token}",
            },
            'json': {
                "limit": 7,
                "mode": "rag",
                "model_name": self._model_name,
                "must_not_sids": [],
                "query": query,
                "sources":["service-public", "travail-emploi"],
                "temperature": self._temperature,
                "with_history": with_history
            },
        }
        response = await asyncio.to_thread(requests.post, **request)
        json = response.json()
        stream_id = json.get('id')
        return stream_id

    def fetch_stream(self, stream_id:int) -> str:
        raw_data:str = ""
        # Create an SSE client
        messages = sseclient.SSEClient(
            url = ALBERT_URL_FETCH_STREAM.format(stream_id = stream_id),
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._token}",
                "Connection": 'keep-alive',
            },
        )

        # Process the stream
        for msg in messages:
            tmp:str = msg.data.encode().decode('unicode_escape').strip('"')
            if tmp == '[DONE]':
                break
            raw_data = "".join([raw_data, tmp])
        return raw_data


    # TODO:
    # keep an eye on the rate limite
    async def complete(self, prompt:Prompt) -> LLMAnswer:
        start_ts:datetime = datetime.now()
        await self._refresh_token()
        stream_id = await self._init_stream(prompt.user)
        result = self.fetch_stream(stream_id = stream_id)
        duration = 0 # is the duration of the api call
        cost = 0.0 # is the cost issued from a api call

        return LLMAnswer(
            name = self.name,
            prompt = prompt,
            text = result, #need refacto see albert doc,
            full_name = self._model_name, # full name of the model?
            timestamp = start_ts,
            duration = duration,
            cost = cost,
        )
