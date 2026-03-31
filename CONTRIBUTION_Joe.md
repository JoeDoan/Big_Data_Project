# Individual Contribution Report — Labs 1–9 (Through Phase 3)
**Name:** Joe Doan
**Role:** Data Pipeline Engineer & Domain Adaptation Lead

---

## Phase 3 Contributions

### 1. BERT vs LLM Evaluation & Pipeline Decision (`evaluate_e2e.py`)
- Designed and ran the **unified end-to-end pipeline evaluation** across 11 CUAD test contracts, testing three extraction approaches:
  - **BERT Risk Clause Detection** (12 clauses): 53.8% accuracy, near-zero recall — BERT's 512-token sentence windows missed clauses spanning multiple paragraphs.
  - **Full-Document LLM Extraction** (V4, 8 entities): 86.3% accuracy — Gemini 2.5 Flash reads the entire contract in one pass.
  - **Hybrid Search Retrieval** (5 configurations): Benchmarked `all-MiniLM-L6-v2` vs `BAAI/bge-small-en-v1.5`, varying top-K and query expansion.
- Made the **architectural decision** to remove BERT from production and replace with full-document LLM extraction, based on evaluation evidence.

### 2. Full-Document LLM Extraction Pipeline (`tools.py`)
- Implemented `extract_risk_clauses_llm()`: passes full contract text (up to 200,000 characters) directly to Gemini 2.5 Flash to scan for 12 critical clause types simultaneously, returning structured JSON with detected status, risk level, exact excerpt, and section reference.
- Implemented `extract_contract_brief()`: V4 chain-of-thought entity extraction for 8 metadata fields (Document Name, Parties, Agreement Date, Effective Date, Expiration Date, Renewal Term, Notice to Terminate, Governing Law) with multi-hop date reasoning.
- **Key insight:** Passing the full document directly to the LLM (leveraging Gemini's 1M-token context window) outperforms chunking + retrieval because chunking fragments clause context across boundary edges.

### 3. Snowflake Chat Persistence (`chat_history.py`)
- Designed and implemented the **Snowflake-backed chat history system** with two tables:
  - `CHAT_SESSIONS`: SESSION_ID, TITLE (LLM-generated), CREATED_AT, UPDATED_AT
  - `CHAT_MESSAGES`: MESSAGE_ID, SESSION_ID, ROLE, CONTENT, RISK_LEVEL, METADATA (JSON-serialized annotations)
- Implemented `save_message()` with automatic annotation serialization — expandable source citations (clause name, risk level, excerpt, section) are stored as JSON in the METADATA column and fully restored on session reload.
- Implemented `generate_session_title()` using Gemini 2.5 Flash to create 3-6 word descriptive titles from the first user message.
- Implemented `delete_session()` for cascading session + message deletion.

### 4. UI Enhancements (`app.py`)
- **Dark/Light Theme Toggle:** Implemented a CSS variable-based theme system with a sidebar toggle switch, applying consistent styles across all components (chat bubbles, glass cards, input fields, tables, file uploaders) in both modes.
- **Chat History Sidebar Panel:** Built a persistent session list showing recent conversations with clickable entries for session switching, 🗑️ delete buttons, and ➕ New Chat functionality.
- **Annotation Persistence:** Ensured that expandable source annotations ([1] 📋 clause_name — Risk Level) are preserved across page reloads by serializing them to Snowflake and deserializing on session load.

### 5. Phase 3 Report (`Phase_3_Report_LexGuard.docx`)
- Authored the complete Phase 3 report documenting all architectural changes, evaluation results, Snowflake schema, and pipeline evolution.

---

## Lab 9 Contributions

### 1. Structured Execution Traces (`agent.py`, `adapted_agent.py`)
- Redesigned both agent functions to return **structured dict responses** containing: `response`, `trace`, `tool_calls`, `retrieval_count`, `risk_level`, and `success`.
- Implemented **per-step timing** in the baseline agent: each Gemini tool call is wrapped with `time.time()` measurements, capturing tool name, arguments, result preview, and elapsed time.
- Implemented **three-phase timing** in the adapted agent: RAG retrieval time, Colab model inference time, and risk calculation time are each individually measured.

### 2. Debug Logging Integration
- Added structured trace entries for every execution step: `start`, `tool_call`, `model_inference`, `response`, `error`, `greeting_filter`, `no_results`, and `timeout`.

### 3. Development Report (`LAB9_REPORT.md`)
- Authored the group development report covering all 4 enhancement areas, deployment method, and how the system extends Phase 2.

---

## Lab 8 Contributions

### 1. Instruction Dataset Generation (`generate_dataset.py`)
- Designed and implemented the **instruction dataset pipeline** using the Gemini API to generate 50 high-quality training examples from the CUAD contract corpus.
- Each example follows the required Lab 8 format with `instruction`, `input`, and `output` fields, producing verbose, step-by-step legal reasoning outputs.

### 2. Domain-Adapted Agent Pipeline (`adapted_agent.py`)
- Designed and implemented the **full PEFT inference pipeline** integrating the fine-tuned Llama-3 adapter with the local RAG store.
- Debugged the critical **prompt format mismatch bug**: the Llama-3 adapter was receiving Mistral `[INST]` tags, causing an infinite generation loop.
- Implemented robust **response parsing** by identifying that Unsloth's tokenizer renders special tokens as plain text.

### 3. FastAPI Colab Server Configuration
- Configured the Google Colab inference server, diagnosed and fixed the Llama-3 GPU deadlock caused by incorrect input format.

### 4. Lab 8 Evaluation (`run_evaluation.py`, `EVALUATION.md`)
- Wrote the evaluation script that ran both agents on all 10 queries and recorded timing and answer-rate metrics.

---

## Previous Lab Contributions (Labs 1–7)
- Built `ingest.py`: complete PDF → Snowflake ingestion pipeline with OCR fallback, dual-write to `LocalStore`, and deterministic UUID chunk IDs.
- Curated and organized the 6 contract PDFs in `./data/` from public SEC/CUAD sources.
- Implemented `get_snowflake_connection()` with auto-provisioning and MFA support.

---

## Links to Commits
- [Initial commit: environment setup](https://github.com/Manan151179/BIG_DATA_LAB6/commit/b6abf4a)
- [Initial commit for Lab 6: LexGuard Agent](https://github.com/Manan151179/BIG_DATA_LAB6/commit/7c4ea6e)

---

## AI Tools Used
- **Antigravity (Google DeepMind)**: Used to debug Llama-3 prompt formatting (Lab 8), implement structured execution traces (Lab 9), build chat history persistence, dark/light theme system, and annotation serialization (Phase 3).
- **Gemini API**: Used to generate instruction dataset training examples, session titles, and full-document contract extraction.

---

## Technical Reflection

The most impactful decision in Phase 3 was abandoning the BERT extraction pipeline in favor of direct full-document LLM input. The evaluation data was unambiguous: BERT's 53.8% accuracy with near-zero recall meant it was essentially useless for detecting clauses that actually exist in contracts. The root cause was architectural — BERT's 512-token window fundamentally cannot capture clauses that span multiple paragraphs or reference earlier sections of the document. Gemini's 1-million-token context window eliminates this problem entirely by processing the full contract in a single pass, achieving 86.3% accuracy. This taught me that sometimes the best engineering decision is to remove complexity rather than add it — the chunking + retrieval pipeline was technically sophisticated but ultimately counterproductive for this specific use case.
