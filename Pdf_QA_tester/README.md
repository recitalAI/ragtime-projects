# Pdf_QA_tester


## Introduction

This project aims to test the generation capabilities of Large Language Models (LLMs) when integrated with a RAG (Retrieval-Augmented Generation) system. The process involves generating questions from a set of PDF documents, retrieving relevant context chunks for the questions, and evaluating the answers provided by LLMs.

## Contributors

- ZAOUG Imad

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

<img src="img/Q_and_A_only.png">

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

#### Implementation
- In the project, the BM25 retriever retrieves relevant chunks of text based on the query terms provided.
- It calculates the relevance score for each chunk using BM25 scoring algorithm.
- The top-ranked chunks are selected for further processing.

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
