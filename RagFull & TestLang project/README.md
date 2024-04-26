# RagFull & TestLang project [RF] [TL]: Ragtime with Llamaindex: Answer & Fact Generation and Evaluation

This project demonstrates an example of Answer & Fact generation and automatic evaluation using Ragtime 🎹. The goal is to test the generation capabilities of Large Language Models (LLMs) when integrated with a Retrieval-Augmented Generation (RAG) system powered by Llamaindex.

## Process Overview

1. **Prepare Your Dataset**:
   - Gather the PDF documents you want to use in your RAG system.
   - Ensure that the dataset is well-organized and accessible.

2. **Create Your Questions**:
   - Generate a set of questions that will serve as prompts for the LLMs.
  
3. **Context Retrieval**:
   - Retrieve relevant context chunks from the same PDF documents.
   - These context chunks provide additional information that can enhance the quality of generated answers.
 
4. **Answer Evaluation**:
   - After the LLMs generate answers, evaluate their quality.
   - This step helps assess how well the LLMs perform in providing accurate and relevant answers.
   


## Create the project
To do so, we first set `PROJECT_NAME='RagFull & TestLang project'` in `main.py` and run the script. It will create several sub folders as well as some 
files. 

Go to the `RagFull & TestLang project` subfolder and set the `PROJECT_NAME` variable in `main.py` to `'RagFull & TestLang project'`.

## Getting Started

1. **Set up your environment**:
- Install necessary Python libraries (e.g., llama_index, dotenv, etc.). (requirement.txt)
- Configure your environment variables (e.g., API keys, file paths).

2. **Set up your Database**:
We place the PDF documents we want to use in the "data1" and "data2" folders.

3. **Create your questions**:
We then prepare a set of questions (16Q) using a JSON question file. If you open the  `questionsTest.json ` file in the `expe/01. Questions` folder, you will see something like this:
```json
[
  {
    "meta": {},
    "items": [
      {
        "meta": {},
        "question": {
          "meta": {},
          "text": "Quand mon contrat d'assurance prend-il effet et combien de temps dure-t-il??"
        },
        "facts": {
          "llm_answer": null,
          "meta": {},
          "items": []
        },
        "chunks": {
          "meta": {},
          "items": []
        },
        "answers": {
          "meta": {},
          "items": []
        }
      },
    },
[...]
```
4. **Notes about files**:
   ## `RAG.py`

Contains utility functions designed to facilitate operations within the RAG system:

- `load_env`: This function loads environment variables from a `.env` file using the `load_dotenv()` function from the `dotenv` module. It also sets the value of the environment variable `'OPENAI_API_KEY'` to an empty string, which is commonly used for storing API keys or sensitive information related to OpenAI services.
- `load-document(list: folders)`: Reads documents stored within folders.
- `create_index(documents)`: Establishes the vector store index.

## `classes.py`

This file introduces two classes:

- `MyRetriever`

This class extends the functionality of a Retriever by incorporating a vector-based retrieval mechanism (`vector_retriever`). It retrieves relevant information based on a given question (`qa.question.text`). The retrieved results are converted into Chunks and added to the question's `chunks` attribute.

- `MyAnswerPptr`

As a subclass of Prompter, this class handles the generation of prompts for the user and system based on the provided question and retrieved chunks. In the `get_prompt` method, it constructs a user prompt with the original question text and a system prompt that includes relevant information from the retrieved chunks. In the `post_process` method, it assigns the text of the language model answer (`cur_obj.llm_answer.text`) to the current object (`cur_obj.text`).


## `main_retrieve.py`

Tasked with obtaining relevant text segments for each generated inquiry, this script utilizes `MyRetriever` and stores the outcomes in a JSON file.

## `main_answer.py`

Dedicated to generating responses to inquiries utilizing a language model. The outputs can be exported in HTML and spreadsheet formats.

## `main_facts_evals.py`

This script generates facts based on the produced answers and automatic evaluation. The findings can be exported in HTML and spreadsheet formats.

## Context Retrieval & Answer generation
1.  **Context Retrieval**:
Relevant context chunks are retrieved using vector index retriever method.
The class  `MyRetriever` in  `classes.py`
```python
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
```
Run  `main_retrieve.py ` to retrieve chunks and add them to a JSON file.
You should be able now to find the result file in the `01. Question` directory in `expe`. The generated file is for instance `questions--16Q_64C_0F_0M_0A_0HE_0AE_2024-04-25_14h25,56.json`. It contains the original file name plus the number of questions in the expe (16Q), of chunks (64 in this case), of facts (0F), of models (0M), of answers (0A), of human evaluations (0HE) and of automatic evaluations (0AE). 

2.  **Answer generation**:
Run  `main_answer.py` to obtain answers from the chosen LLMs ("gpt-3.5-turbo" in our case). Ensure dependencies are installed and the environment is set.
You should be able now to find the result file in the `01. Answer` directory in `expe`.

The HTML export is done with:
`expe.export_to_html(json_path=FOLDER_ANSWERS / 'questions--16Q_64C_0F_1M_16A_0HE_0AE_2024-04-26_01h55,36.json')`
Go to `expe/02. Answers` to open the generated file.

The HTMl export looks like this:
<img src="Screenshots/Screenshot 2024-04-26 105613.png">

## Answer evaluation
Before starting, review the generated responses manually by modifying the JSON file. Identify entries where "human" is not specified (null) and rate each response accordingly (In our case we assigned score 1 for all the answers).


1. **Fact Generation**: First uncomment relevant code in `main_fact_evals.py` to generate facts. Don't forget to adjust the path to the JSON file generated by  `main_answer.py `.

    ```python
    generators.gen_Facts(folder_in=FOLDER_ANSWERS, folder_out=FOLDER_FACTS, json_file='questions--16Q_64C_0F_1M_16A_0HE_0AE_2024-04-26_01h55,36.json',
                       llm_names=['gpt-3.5-turbo'], prompter=PptrFactsFRv2())
    ```

2. **Evaluation**: Then Uncomment relevant lines in `main_fact_evals.py` to evaluate the generated facts.  Don't forget again to adjust the path to the JSON file generated by  `main_facts_evals.py `..

    ```python
    generators.gen_Evals(folder_in=FOLDER_FACTS, folder_out=FOLDER_EVALS, 
                      json_file='questions--16Q_64C_81F_1M_16A_16HE_0AE_2024-04-26_02h16,44.json',
                       llm_names=['gpt-3.5-turbo'], prompter=PptrEvalFRv2())
    ```

3. **HTML and XLSX export**: The HTML and XLSX versions of the results are done with: 

    ```python
   expe.export_to_html(json_path=FOLDER_EVALS / "questions--16Q_64C_81F_1M_16A_16HE_12AE_2024-04-26_02h18,45.json")
   expe.export_to_spreadsheet(json_path=FOLDER_EVALS / "questions--16Q_64C_81F_1M_16A_16HE_12AE_2024-04-26_02h18,45.json",
                             template_path=FOLDER_SST_TEMPLATES/'spreadsheet_rich_template.xlsx')
    ```
The HTML output after an Eval shows, on top of the Facts and the Answers, the Evaluations, starting with Hallus:
<img src="Screenshots/Screenshot 2024-04-26 105538.png">

## TestLang: running tests with prompts translated from French to English to see if it has an impact
To visualize the results, there are HTML files in both "02. Answers" and "04. Evals" directories for both cases (French an English prompts).
1. **French prompts**:
- Chunks andAnswers : [Here](https://github.com/Dahbani1/ragtime-projects/blob/main/RagFull%20%26%20TestLang%20project/expe/02.%20Answers/questions--16Q_64C_0F_1M_16A_0HE_0AE_2024-04-25_15h33%2C02.html)
- Facts and Evaluation: [Here](https://github.com/Dahbani1/ragtime-projects/blob/main/RagFull%20%26%20TestLang%20project/expe/02.%20Answers/questions--16Q_64C_0F_1M_16A_0HE_0AE_2024-04-26_01h58%2C14.html)
2. **English prompts**:
- Chunks andAnswers : [Here](https://github.com/Dahbani1/ragtime-projects/blob/main/RagFull%20%26%20TestLang%20project/expe/04.%20Evals/questions--16Q_64C_81F_1M_16A_16HE_12AE_2024-04-26_02h23%2C37.html)
- Facts and Evaluation: [Here](https://github.com/Dahbani1/ragtime-projects/blob/main/RagFull%20%26%20TestLang%20project/expe/04.%20Evals/questions--16Q_64C_92F_1M_16A_16HE_14AE_2024-04-25_20h25%2C12.html)

### Observations
The responses generated from prompts in both French and English intersect but are not identical. Additionally, we can notice that Facts change, thus altering the evaluation of the responses generated by our RAG system since the criteria differ.

## Opportunities for improvements:
- Detecting evaluation not equal to 1 and fixing the facts.
- Run answer generation, fact generation and evaluation with different LLMs to see their influence [JJG] [Contribute](CONTRIBUTING.md)
