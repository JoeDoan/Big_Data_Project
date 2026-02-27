# LexGuard: The Neuro-Symbolic Compliance Auditor

**Team Members:**
* **Manan Koradiya** 
* **Joe Doan** 
* **Ruixuan Hou** 
* **Aditya Naredla** 

LexGuard is an advanced Multi-Agent Retrieval-Augmented Generation (RAG) system designed to extract critical metadata from complex legal contracts. Leveraging state-of-the-art embedding models and LLMs, LexGuard processes documents dynamically to extract information accurately and present it interactively.

---

## 1. Problem Statement
Standard Large Language Models (LLMs) suffer from stochastic hallucinations, making them unreliable for high-stakes legal compliance. Reviewing contracts for regulatory violations (e.g., Housing Laws, GDPR) currently requires expensive manual labor because AI tools cannot be trusted to strictly follow logic.

**Objective:**
To build "LexGuard," a neuro-symbolic system that decouples "Reading" (GenAI) from "Reasoning" (Knowledge Graphs). This ensures that all compliance decisions are grounded in explicit symbolic rules, providing the interpretability and trustworthiness required by the legal industry.

---

## 2. System Architecture
*(Note: Upload the image from your design folder here)*
![System Architecture](docs/system_architecture.png)

**Pipeline Overview:**
1.  **Ingestion:** PDF Contracts uploaded to Snowflake Internal Stages.
2.  **Extraction (Neural):** LLM (via Snowpark/Cortex) extracts key clauses into structured text.
3.  **Validation (Symbolic):** Python logic (NetworkX) validates clauses against a "Ground Truth" Knowledge Graph.
4.  **Reporting:** Streamlit Dashboard displays a "Pass/Fail" report with specific rule citations.

### Codebase Structure

- `app/app.py`: The Streamlit application serving as the UI. Run this to interact with LexGuard locally.
- `python/pipeline.py`: The core RAG pipeline implementation (`LexGuardPipeline`). Handles document loading, indexing, searching, and metadata extraction.
- `python/evaluate.py`: The automated evaluation script comparing extracted values vs. CSV annotations using an LLM Judge.
- `python/database.py`: Connects to Snowflake for conversational memory storage.
- `python/rules_engine.py`: Encapsulates any predefined rules or heuristic checks for field values.
- `python/prep_finetune.py` & `train_colab.ipynb`: Utilities to generate fine-tuning data and train custom LLM extraction models.
- `sql/setup_snowflake.sql`: SQL scripts to build the Snowflake database, schema, and tables.
- `.env.example`: A template for the necessary environment variables (`GROQ_API_KEY`, `GEMINI_API_KEY`, `SNOWFLAKE_` credentials).

---

## 3. Getting Started

### 3.1 Requirements and Setup

It is highly recommended to use a virtual environment or Conda environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

### 3.2 Environment Variables

Make a copy of the example environment file:
```bash
cp .env.example .env
```
Fill in the respective API keys obtained from Groq and Google AI Studio into the `.env` file. You may optionally include your Snowflake credentials to enable chat-saving mechanisms.

### 3.3 Running the App

Start the Streamlit web interface:

```bash
streamlit run app/app.py
```

### 3.4 Running Evaluations

If you want to evaluate the pipeline's performance against a dataset (such as CUAD_v1):

```bash
python python/evaluate.py
```
This script will sample a document out of your test set, orchestrate extraction, and use the LLM-as-a-judge model to verify accuracy.

---

## 4. Data Sources & References

### **Datasets**
* **[CUAD (Contract Understanding Atticus Dataset)](https://www.atticusprojectai.org/cuad):** Expert-labeled legal contracts for training/evaluating clause extraction.
* **[GSA Lease Documents](https://catalog.data.gov/dataset/lease-documents):** Real-world federal lease agreements used for system stress-testing.
* **[LEDGAR](https://huggingface.co/datasets/lex_glue):** Annotated legal provisions for clause classification.

### **Research Papers (NeurIPS 2025)**
* **[HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation](https://arxiv.org/abs/2503.21322)** - *Justification for using graph structures over vector-only RAG.*
* **[Alleviating Hallucinations in LLMs through Multi-Model Contrastive Decoding](https://neurips.cc/virtual/2025/poster/118154)** - *Basis for our hallucination detection layer.*
* **[From Semantics to Symbols: A Two-Stage Framework for Deconstructing LLM Reasoning](https://neurips.cc/virtual/2025/125206)** - *The foundational architecture for our Neuro-Symbolic approach.*
