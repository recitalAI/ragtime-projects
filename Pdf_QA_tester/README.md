# RagFull[RF]

## Introduction

This project aims to test the generation capabilities of Large Language Models (LLMs) when integrated with a RAG (Retrieval-Augmented Generation) system. The process involves generating questions from a set of PDF documents, retrieving relevant context chunks for the questions, and evaluating the answers provided by LLMs.

## Contributors

- ZAOUG Imad

## Setup

1. **Database Setup**: Place the PDF documents you want to use in the "pdf" folder.

2. **Question Generation**: Run `question_generation.py` to generate questions. You can test on the provided PDFs or your own.

    - The script `question_generation.py` utilizes the function `question_gen(name:str)` in `Rag.py` to generate questions.
    - The function randomizes the question order for evaluation purposes.

3. **Context Retrieval**: Relevant context chunks are retrieved using Okapi BM25 and index retriever methods. The top 10 results are selected and merged.

    - The class `MyRetriever` in `Rag.py` combines BM25 and vector retrievers to retrieve context chunks.

4. **Adding Chunks to JSON**: Run `retrieve_chunk.py` to add the chunks into the JSON file.

5. **Answer Generation**: Run `main_answer_generation.py` to obtain answers from the chosen LLMs (e.g., "gpt-4" and "gpt-3.5-turbo"). Ensure dependencies are installed and the environment is set.

6. **Visualization**: Visualize the answers by opening the generated HTML file located in `expe/02. Answers` folder.

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

## Datasets

- **Transformers for Machine Learning_ A Deep Dive.pdf**: course material.
- **Docs**: Example [dataset](https://storage.recital.ai/s/ZnIx.GWJqg2ZXgGpPq4o).

To test with the course material, ensure to specify the PDF path in `question_generation.py`:

```python
if __name__ == '__main__':
    path = "pdf" #here

    while True:
        try:
            num_quest = int(input("Enter the number of questions you want to test the model on: "))
            break  # Exit the loop if the input is successfully converted to an integer
        except ValueError:
            print("Please enter an integer.")

    # Call the main function with the validated integer value for 'num_quest'
    main(path=path, num_quest=num_quest)
```

In the `eval_question` function, set the `recursive` parameter to `False` so the model only selects PDFs from the folder, excluding those in subfolders:

```python
eval_questions = question_gen(name=path, recursive=False)
```

To test with the example dataset, ensure to specify the PDF path in `question_generation.py`:

```python
if __name__ == '__main__':
    path = "pdf/docs" #here

    while True:
        try:
            num_quest = int(input("Enter the number of questions you want to test the model on: "))
            break  # Exit the loop if the input is successfully converted to an integer
        except ValueError:
            print("Please enter an integer.")

    # Call the main function with the validated integer value for 'num_quest'
    main(path=path, num_quest=num_quest)
```

In the `eval_question` function, leave it as it is because the `recursive` parameter by default is set to `True` to include all PDF files in the subfolders:

```python
eval_questions = question_gen(name=path)
```

## Results

To visualize the results, there is an HTML file in both "02. Answers" and "04. Evals" directories for both cases.

- Question answers for the course case: [Link](https://github.com/ImadZaoug/ragtime-projects/blob/main/Pdf_QA_tester/expe/02.%20Answers/questions--10Q_170C_0F_2M_20A_0HE_0AE_2024-04-22_09h29%2C27.html)

- Question answers for the Example dataset case: [Link](https://github.com/ImadZaoug/ragtime-projects/blob/main/Pdf_QA_tester/expe/02.%20Answers/questions--30Q_600C_0F_2M_60A_0HE_0AE_2024-04-24_14h17%2C45.html)

- Evals for the course case: [Link](https://github.com/ImadZaoug/ragtime-projects/blob/main/Pdf_QA_tester/expe/04.%20Evals/questions--10Q_170C_72F_2M_20A_20HE_20AE_2024-04-22_10h01%2C31.html)

- Evals for the Example dataset case: [Link](https://github.com/ImadZaoug/ragtime-projects/blob/main/Pdf_QA_tester/expe/04.%20Evals/questions--30Q_600C_174F_2M_60A_60HE_59AE_2024-04-24_14h29%2C52.html)

## Note

You can add other LLMs for evaluation by modifying the code accordingly and ensuring dependencies are met.

## Retriever advanced explanation

### BM25 Retriever

#### Overview
BM25 (Best Matching 25) is a probabilistic information retrieval model based on the probabilistic relevance framework. It's an extension of the TF-IDF (Term Frequency-Inverse Document Frequency) model. 

#### Functionality
1. **Bag-of-Words Approach**: BM25 treats each document as a bag of words, ignoring the order in which they appear. It calculates relevance based on the presence of query terms in each document.
   
2. **Ranking Documents**: It ranks documents based on the similarity between the query and the document. The relevance score is determined by considering the term frequency, document length, and inverse document frequency.

3. **Scalability**: BM25 is scalable to large collections of documents and performs well in real-world scenarios.

### Vector Retriever

#### Overview
The Vector Retriever utilizes vector representations of documents and queries to find relevant documents. It operates on the principle of similarity between vectors in a high-dimensional space.

#### Functionality
1. **Vector Representation**: Each document and query is represented as a vector in a high-dimensional space. These vectors capture the semantic meaning of the text.

2. **Similarity Calculation**: Similarity between documents and queries is computed using metrics like cosine similarity or Euclidean distance. Documents with vectors closest to the query vector are considered most relevant.

3. **Flexibility**: Vector retrievers are flexible and can capture complex relationships between words and documents.

### Comparison
- **BM25**: Relies on term frequency and document length, suitable for bag-of-words representation.
- **Vector Retriever**: Captures semantic meaning using vector representations, suitable for capturing complex relationships and semantics in text.

In this project, both retrievers are used in conjunction to retrieve relevant chunks of text for further analysis and evaluation by the LLMs.
