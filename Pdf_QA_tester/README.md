# Project Name

## Introduction

This project aims to test the generation capabilities of Large Language Models (LLMs) when integrated with a RAG (Retrieval-Augmented Generation) system. The process involves generating questions from a set of PDF documents, retrieving relevant context chunks for the questions, and evaluating the answers provided by LLMs.

## Setup

1. **Database Setup**: Place the PDF documents you want to use in the "pdf" folder.

2. **Question Generation**: Run `question_generation.py` to generate questions. You can test on the provided PDFs or your own.

3. **Context Retrieval**: Relevant context chunks are retrieved using Okapi BM25 and index retriever methods. The top 10 results are selected and merged.

4. **Adding Chunks to JSON**: Run `retrieve_chunk.py` to add the chunks into the JSON file.

5. **Answer Generation**: Run `main_answer_generation.py` to obtain answers from the chosen LLMs (e.g., "gpt-4" and "gpt-3.5-turbo"). Ensure dependencies are installed and environment is set.

6. **Visualization**: Visualize the answers by opening the generated HTML file located in `expe/01. Questions` folder.

## Manual Human Evaluation

Manually evaluate the generated answers by editing the JSON file. Locate entries with `"human": null` and assign a score for each answer.

## Fact Generation and Evaluation

1. **Fact Generation**: Uncomment relevant code in `main_fact_evals.py` to generate facts. Adjust the path to the JSON file.

2. **Evaluation**: Uncomment relevant lines in `main_fact_evals.py` to evaluate the generated facts. Ensure the JSON path is correct.

3. **Export Results**: Generate HTML and XLSX versions of the results using the provided commands. Make sure to use the correct JSON path.

## How to Run

1. Install the required dependencies:
   ```
   pip install -r requirement.txt
   ```

2. Follow the steps outlined above for question generation, context retrieval, answer generation, manual evaluation, fact generation, and evaluation.

## Note

You can add other LLMs for evaluation by modifying the code accordingly and ensuring dependencies are met.

## Contributors

- [Your Name]
- [Additional Contributors if any]

## License

This project is licensed under [License Name]. See the LICENSE file for details.
