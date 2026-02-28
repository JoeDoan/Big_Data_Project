# Team Contributions for Phase 2

## Project: LexGuard - The Neuro-Symbolic Compliance Auditor

| Member | Contribution Description | Contribution % |
| :--- | :--- | :--- |
| **Manan Koradiya** | Developed the core **RAG Pipeline & Embedding Strategy** (`pipeline.py`). Implemented the hybrid search configuration (FAISS + BM25), the multi-agent `gemini-2.5-flash` targeted extraction batches, and text chunking logic (Fixed, Page, Semantic). | 25% |
| **Joe Doan** | Built the **Streamlit Application & Integration** (`app/app.py`). Designed the UI for document upload, evidence display, the interactive RAG Co-Pilot chat, and integrated the rule engine's automated risk assessment dashboard. | 25% |
| **Ruixuan Hou** | Created the **Evaluation Engine & Symbolic Rules logic** (`evaluate.py`, `rules_engine.py`). Crafted the LLM-as-a-judge workflow using Groq to score extraction accuracy against the CUAD dataset ground truth, and implemented the deterministic Python risk triggers. | 25% |
| **Aditya Naredla** | Engineered the **Snowflake Cloud Data Pipeline & Data Prep** (`database.py`, `sql/setup_snowflake.sql`, `prep_finetune.py`). Established Snowflake connection workflows to safely log conversational data, store chunk JSONs in VARIANT columns, and mapped raw PDF data to `.jsonl` for LLM finetuning. | 25% |

*Note: All contributions are supported by commit history mapping to the architecture, pipeline integrations, and system validation tests detailed in the main Phase 2 Report.*
