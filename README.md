# LexGuard: The Neuro-Symbolic Compliance Auditor

**Team Members:**
* **Manan Koradiya** 
* **Joe Doan** 
* **Ruixuan Hou** 
* **Aditya Naredla** 

---

## 1. Problem Statement
Standard Large Language Models (LLMs) suffer from stochastic hallucinations, making them unreliable for high-stakes legal compliance. Reviewing contracts for regulatory violations (e.g., Housing Laws, GDPR) currently requires expensive manual labor because AI tools cannot be trusted to strictly follow logic.

**Objective:**
To build "LexGuard," a neuro-symbolic system that decouples "Reading" (GenAI) from "Reasoning" (Knowledge Graphs). This ensures that all compliance decisions are grounded in explicit symbolic rules, providing the interpretability and trustworthiness required by the legal industry.

---

## 2. System Architecture
*(Note: Upload the image from your design folder here)*
`![System Architecture](docs/system_architecture.png)`

**Pipeline Overview:**
1.  **Ingestion:** PDF Contracts uploaded to Snowflake Internal Stages.
2.  **Extraction (Neural):** LLM (via Snowpark/Cortex) extracts key clauses into structured text.
3.  **Validation (Symbolic):** Python logic (NetworkX) validates clauses against a "Ground Truth" Knowledge Graph.
4.  **Reporting:** Streamlit Dashboard displays a "Pass/Fail" report with specific rule citations.

---

## 3. Data Sources & References

### **Datasets**
* **[CUAD (Contract Understanding Atticus Dataset)](https://www.atticusprojectai.org/cuad):** Expert-labeled legal contracts for training/evaluating clause extraction.
* **[GSA Lease Documents](https://catalog.data.gov/dataset/lease-documents):** Real-world federal lease agreements used for system stress-testing.
* **[LEDGAR](https://huggingface.co/datasets/lex_glue):** Annotated legal provisions for clause classification.

### **Research Papers (NeurIPS 2025)**
* **[HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation](https://arxiv.org/abs/2503.21322)** - *Justification for using graph structures over vector-only RAG.*
* **[Alleviating Hallucinations in LLMs through Multi-Model Contrastive Decoding](https://neurips.cc/virtual/2025/poster/118154)** - *Basis for our hallucination detection layer.*
* **[From Semantics to Symbols: A Two-Stage Framework for Deconstructing LLM Reasoning](https://neurips.cc/virtual/2025/125206)** - *The foundational architecture for our Neuro-Symbolic approach.*
