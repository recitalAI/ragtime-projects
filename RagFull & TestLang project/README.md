# RagFull & TestLang project [RF] [TL]
This project shows first an example of Answer & Fact generation and automatic Evaluation in Ragtime 🎹 to test the generation capabilities of Large Language Models (LLMs) when integrated with a RAG (Retrieval-Augmented Generation) system powered by Llamaindex. The process involves 4 main steps as follow:

- Prepare your dataset: prepare the PDF documents you want to use in your RAG system.
- Create your questions: These questions serve as prompts for the LLMs.
- Context Retrieval: You retrieve relevant context chunks from the same PDF documents. These context chunks provide additional information that can help the LLMs generate better answers.
- Answer Evaluation: After the LLMs generate answers, you evaluate their quality. This step helps assess how well the LLMs perform in providing accurate and relevant answers.

## Create the project
To do so, we first set `PROJECT_NAME='RagFull &TestLang project'` in `main.py` and run the script. It will create several sub folders as well as some 
files. 

Go to the `google_nq` subfolder and set the `PROJECT_NAME` variable in `main.py` to `'google_nq'`.
