# In this file you define the different classes used for your experiments within Ragtime :
# - an optional Retriever if you first have to get chunks
# - a Prompter for Answer generation
# - an optional Prompter for Fact generation
# - an optional Prompter for Eval generation

from typing import Optional
from ragtime.expe import QA, Chunks, Prompt, Question, WithLLMAnswer, Chunk
from ragtime.generators import Prompter, Retriever
from llama_index.core.retrievers import VectorIndexRetriever
from pydantic import BaseModel



class MyRetriever(Retriever):
    vector_retriever: VectorIndexRetriever

    class Config:
        arbitrary_types_allowed = True

    def retrieve(self, qa: QA): 
        result = self.vector_retriever.retrieve(qa.question.text)
        # Convert retrieved results to Chunks and add them to qa.chunks
        for k in result:
            chunk = Chunk(meta = {"score" : k.score, "node Id" : k.node_id}, text=k.text)
            qa.chunks.append(chunk)

class MyAnswerPptr(Prompter):
    def get_prompt(self, question:Question, chunks:Optional[Chunks] = None) -> Prompt: 
            prompt = Prompt()

            # Use the original question for the user prompt
            prompt.user = f"{question.text}"

            # If chunks are available, incorporate them into the system prompt (replace as needed)
            if chunks:
                system_prompt = "Here's what I found relevant to your question:\n"
                for chunk in chunks:
                    system_prompt += f"- {chunk.text[:100]}\n"
                    prompt.system = system_prompt
            else:
                prompt.system = "I couldn't find anything relevant."

            return prompt
    
    def post_process(self, qa:QA=None, cur_obj:WithLLMAnswer=None):
       cur_obj.text = cur_obj.llm_answer.text