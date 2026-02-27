# LexGuard

LexGuard is an advanced Multi-Agent Retrieval-Augmented Generation (RAG) system designed to extract critical metadata from complex legal contracts. Leveraging state-of-the-art embedding models and LLMs, LexGuard processes documents dynamically to extract information accurately and present it interactively.

## Features

- **Document Ingestion:** Works with standard document formats (`.txt` and `.pdf`). Features multiple chunking strategies (Fixed length, Page-based, Semantic paragraph).
- **RAG Architecture:** Employs Dense Retrieval (via FAISS and `nomic-embed-text-v1.5`), Sparse Retrieval (BM25), and Cross-Encoder Reranking (`ms-marco-MiniLM-L-6-v2`) for unparalleled context retrieval accuracy.
- **Multi-Agent Extraction:** Divides metadata fields into semantic batches (Core Terms, Covenants & Terms, IP & Licensing, Liability & Audit) to perform targeted RAG extraction using high-quality LLMs like Llama or Gemini via Groq API.
- **Automated Evaluation (LLM-as-a-Judge):** Features a comprehensive evaluation pipeline against ground-truth CSV datasets, utilizing a secondary LLM to judge semantic matches instead of strict exact matching.
- **Interactive UI:** A Streamlit web application providing a chat-like interface allowing users to query documents, view extracted parameters dynamically, and manage document chunks.
- **Chat History:** Capable of persisting interactions using a Snowflake database backend.

## Structure

- `app/app.py`: The Streamlit application serving as the UI. Run this to interact with LexGuard locally.
- `python/pipeline.py`: The core RAG pipeline implementation (`LexGuardPipeline`). Handles document loading, indexing, searching, and metadata extraction.
- `python/evaluate.py`: The automated evaluation script comparing extracted values vs. CSV annotations using an LLM Judge.
- `python/database.py`: Connects to Snowflake for conversational memory storage.
- `python/rules_engine.py`: Encapsulates any predefined rules or heuristic checks for field values.
- `python/prep_finetune.py` & `train_colab.ipynb`: Utilities to generate fine-tuning data and train custom LLM extraction models.
- `sql/setup_snowflake.sql`: SQL scripts to build the Snowflake database, schema, and tables.
- `.env.example`: A template for the necessary environment variables (`GROQ_API_KEY`, `GEMINI_API_KEY`, `SNOWFLAKE_` credentials).

## Getting Started

### 1. Requirements and Setup

It is highly recommended to use a virtual environment or Conda environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Make a copy of the example environment file:
```bash
cp .env.example .env
```
Fill in the respective API keys obtained from Groq and Google AI Studio into the `.env` file. You may optionally include your Snowflake credentials to enable chat-saving mechanisms.

### 3. Running the App

Start the Streamlit web interface:

```bash
streamlit run app/app.py
```

### 4. Running Evaluations

If you want to evaluate the pipeline's performance against a dataset (such as CUAD_v1):

```bash
python python/evaluate.py
```
This script will sample a document out of your test set, orchestrate extraction, and use the LLM-as-a-judge model to verify accuracy.

## Environment Variables
- `GROQ_API_KEY`
- `GEMINI_API_KEY`
- `SNOWFLAKE_USER` (Optional)
- `SNOWFLAKE_PASSWORD` (Optional)
- `SNOWFLAKE_ACCOUNT` (Optional)
- `SNOWFLAKE_DATABASE` (Optional)
- `SNOWFLAKE_SCHEMA` (Optional)
