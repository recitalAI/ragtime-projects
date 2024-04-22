import logging
import sys
import pandas as pd

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from Rag import question_gen
from ragtime.expe import QA, Expe
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
def main() :
    

    eval_questions = question_gen(name="pdf")
    exper = Expe()
    for i in range(10):
        # Create a new QA object for each question
        Questions = QA()
        Questions.question.text = eval_questions[i]
        exper.append(Questions)

    exper.save_to_json(path = Path("expe/01. Questions/questions.json"))

    



if __name__ == '__main__':
    # Call the main function to get the value for 'name'
    main() 