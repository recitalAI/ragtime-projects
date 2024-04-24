# Pdf_QA_tester


## Introduction

This project aims to test the generation capabilities of Large Language Models (LLMs) when integrated with a RAG (Retrieval-Augmented Generation) system. The process involves generating questions from a set of PDF documents, retrieving relevant context chunks for the questions, and evaluating the answers provided by LLMs.

## Contributors

- ZAOUG Imad

Sure, here's a concise and well-explained README for all the files you provided:

## Files

### `rag.py`

This file contains utility functions for working with the RAG system:

- `read_doc(name: str)`: Reads documents from a directory.
- `question_gen(name: str)`: Generates evaluation questions from documents.
- `nodes_cr(name: str)`: Creates nodes (text chunks) from documents.
- `index_cr(nodes)`: Creates a vector store index from the nodes for efficient retrieval.

### `classes.py`

This file defines two custom classes:

- `MyRetriever`: Retrieves relevant chunks of text from the document corpus based on a query, combining results from BM25 and vector similarity retrieval.
- `MCQAnsPptr`: Generates prompts for the language model, including the question and relevant context from retrieved chunks.

### `question_generation.py`

This script generates evaluation questions from the document corpus and saves them as a JSON file.

### `retrieve_chunks.py`

This script retrieves relevant chunks of text for each generated question using `MyRetriever` and saves the results as a JSON file.

### `main_answer_generation.py`

This script generates answers to the questions using a language model. It initializes the project, creates instances of `MyRetriever` and `MCQAnsPptr`, retrieves relevant chunks, generates prompts, and passes them to the language model. The results are exported to HTML and spreadsheet formats.

### `main_facts_evals.py`

This script generates facts and evaluations based on the generated answers. It initializes the project and contains commented-out sections for generating facts and evaluations using specific prompters. The results can be exported to HTML and spreadsheet formats.

## Usage

1. Set up the required dependencies and environment variables.
2. Run `question_generation.py` to generate evaluation questions.
3. Run `retrieve_chunks.py` to retrieve relevant chunks for the questions.
4. Run `main_answer_generation.py` to generate answers using a language model.
5. (Optional) Run `main_facts_evals.py` to generate facts and evaluations based on the answers.

Note: Adjust file paths and configurations as needed for your setup.



## Setup

1. **Database Setup**: Place the PDF documents you want to use in the "pdf" folder.

2. **Question Generation**: Run `question_generation.py` to generate questions. You can test on the provided PDFs or your own.

    - The script `question_generation.py` utilizes the function `question_gen(name:str)` in `Rag.py` to generate questions.
    - The function randomizes the question order for evaluation purposes.

    ```python
    def question_gen(name:str):
        documents = read_doc(name)
        data_generator = DatasetGenerator.from_documents(documents)
        eval_questions = data_generator.generate_questions_from_nodes()
        random.shuffle(eval_questions)
        return eval_questions
    ```

3. **Context Retrieval**: Relevant context chunks are retrieved using Okapi BM25 and index retriever methods. The top 10 results are selected and merged.

    - The class `MyRetriever` in `Rag.py` combines BM25 and vector retrievers to retrieve context chunks.

    ```python
    class MyRetriever(Retriever, BaseRetriever):
        vector_retriever: VectorIndexRetriever
        bm25_retriever: BM25Retriever

        def _retrieve(self, query: str, indexer=None, similarity_top_k: Optional[int] = None, **kwargs) -> list:
            bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
            vector_nodes = self.vector_retriever.retrieve(query, **kwargs)

            # combine the two lists of nodes
            all_nodes = []
            node_ids = set()
            for n in bm25_nodes + vector_nodes:
                if n.node.node_id not in node_ids:
                    all_nodes.append(n)
                    node_ids.add(n.node.node_id)
            return all_nodes

        def retrieve(self, qa: QA, **kwargs):
            result = self._retrieve(qa.question.text, **kwargs)
            for r in result:
                chunk = Chunk()
                chunk.text , chunk.meta = r.text , {"score":r.score, "Node id" : r.node_id }
                # Check if the chunk already exists in qa.chunks
                existing_chunk = next((c for c in qa.chunks if c.text == chunk.text and c.meta == chunk.meta), None)
                if existing_chunk is None:
                    qa.chunks.append(chunk)
    ```

4. **Adding Chunks to JSON**: Run `retrieve_chunk.py` to add the chunks into the JSON file.

5. **Answer Generation**: Run `main_answer_generation.py` to obtain answers from the chosen LLMs (e.g., "gpt-4" and "gpt-3.5-turbo"). Ensure dependencies are installed and the environment is set.

    - Modify the code to include additional LLMs if desired.

    ```python
    generators.gen_Answers(folder_in=FOLDER_QUESTIONS, folder_out=FOLDER_ANSWERS,
                            json_file='questions--10Q_170C_0F_0M_0A_0HE_0AE_2024-04-22_08h56,06.json',
                            prompter=MCQAnsPptr(), b_missing_only=True,
                            llm_names=["gpt-4", "gpt-3.5-turbo"],retriever = MyRetriever(vector_retriever=vector_retriever,bm25_retriever=bm25_retriever))
    ```

6. **Visualization**: Visualize the answers by opening the generated HTML file located in `expe/02. Answers` folder.

<img src="img/ Q_and_A_only.png">

## Manual Human Evaluation

Manually evaluate the generated answers by editing the JSON file. Locate entries with `"human": null` and assign a score for each answer.

## Fact Generation and Evaluation

1. **Fact Generation**: Uncomment relevant code in `main_fact_evals.py` to generate facts. Adjust the path to the JSON file.

    ```python
    generators.gen_Facts(folder_in=FOLDER_ANSWERS, folder_out=FOLDER_FACTS, json_file='questions--10Q_170C_0F_2M_20A_0HE_0AE_2024-04-22_09h26,25.json',
                         llm_names=['gpt-4'], prompter=PptrFactsFRv2())
    ```

2. **Evaluation**: Uncomment relevant lines in `main_fact_evals.py` to evaluate the generated facts. Ensure the JSON path is correct.

    ```python
    generators.gen_Evals(folder_in=FOLDER_FACTS, folder_out=FOLDER_EVALS, 
                         json_file='questions--10Q_170C_72F_2M_20A_20HE_0AE_2024-04-22_09h43,16.json',
                         llm_names=['gpt-4'], prompter=PptrSimpleEvalFR())
    ```

3. **Export Results**: Generate HTML and XLSX versions of the results using the provided commands. Make sure to use the correct JSON path.

    ```python
    expe.export_to_html(json_path=FOLDER_EVALS / "questions--10Q_170C_72F_2M_20A_20HE_20AE_2024-04-22_09h58,19.json")
    expe.export_to_spreadsheet(json_path=FOLDER_EVALS / "questions--10Q_170C_72F_2M_20A_20HE_20AE_2024-04-22_09h58,19.json",
                               template_path=FOLDER_SST_TEMPLATES/'spreadsheet_rich_template.xlsx')
    ```
The HTMl export for the test:

<img src="img/ Q_and_A_with_C_F_E.png">

## How to Run

1. Install the required dependencies:
   ```
   pip install -r requirement.txt
   ```

2. Follow the steps outlined above for question generation, context retrieval, answer generation, manual evaluation, fact generation, and evaluation.

# Datasets:

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
