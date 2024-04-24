import logging
import sys
import pandas as pd

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from Rag import question_gen, read_doc
from ragtime.expe import QA, Expe
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
def main(path : Path, num_quest : int) :
    

    eval_questions = question_gen(name=path)
    exper = Expe()
    for i in range(num_quest):
        # Create a new QA object for each question
        Questions = QA()
        Questions.question.text = eval_questions[i]
        exper.append(Questions)

    exper.save_to_json(path = Path("expe/01. Questions/questions.json"))

    
if __name__ == '__main__':
    path = "pdf/docs/Conditions générales"

    while True:
        try:
            num_quest = int(input("Indiquez le nombre de questions sur lesquelles vous souhaitez tester le modèle : "))
            break  # Exit the loop if the input is successfully converted to an integer
        except ValueError:
            print("Veuillez entrer un nombre entier.")

    # Call the main function with the validated integer value for 'num_quest'
    main(path=path, num_quest=num_quest)