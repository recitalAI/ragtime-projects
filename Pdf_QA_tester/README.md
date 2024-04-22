# Pdf_QA_tester


## Introduction

This project aims to test the generation capabilities of Large Language Models (LLMs) when integrated with a RAG (Retrieval-Augmented Generation) system. The process involves generating questions from a set of PDF documents, retrieving relevant context chunks for the questions, and evaluating the answers provided by LLMs.

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

## How to Run

1. Install the required dependencies:
   ```
   pip install -r requirement.txt
   ```

2. Follow the steps outlined above for question generation, context retrieval, answer generation, manual evaluation, fact generation, and evaluation.

## Note

You can add other LLMs for evaluation by modifying the code accordingly and ensuring dependencies are met.

## Contributors

- ZAOUG Imad
