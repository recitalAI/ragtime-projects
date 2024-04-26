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
- Install necessary Python libraries (e.g., llama_index, dotenv, etc.).
- Configure your environment variables (e.g., API keys, file paths).

2. **Set up your Database**:
We place the PDF documents you want to use in the "data1" and "data2" folders.

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
...
```
## Context Retrieval & Answer generation



