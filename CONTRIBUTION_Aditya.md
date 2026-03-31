# Individual Contribution Report — Labs 1–9 (Through Phase 3)
**Name:** Aditya Naredla
**Role:** Storage Architect & Evaluation Engineer

---

## Phase 3 Contributions

### 1. Evaluation Support & Hybrid Search Benchmarking
- Supported the hybrid search retrieval evaluation by providing the `LocalStore` infrastructure used in `evaluate_hybrid.py` for testing 5 different configurations (varying embedding models, top-K, and query expansion).
- Maintained the `local_store.py` module that powers the hybrid search fallback path (FAISS dense search + BM25 sparse search + cross-encoder reranking).

### 2. Monitoring Module Maintenance (`monitor.py`)
- Continued maintaining the `QueryMetrics` dataclass and `MetricsCollector` class to ensure compatibility with the Phase 3 UI changes (dark/light theme, chat history integration).
- Analytics dashboard in the sidebar continues to track per-query performance across sessions.

### 3. Training & Model Evaluation
- The PEFT-trained Llama-3 adapter (`doandune/LexGuard-llama3-Risk-Adapter`) on HuggingFace Hub remains available for the adapted pipeline comparison.
- Contributed to the evaluation design that informed the decision to replace BERT with full-document LLM extraction.

---

## Lab 9 Contributions

### 1. Monitoring Module (`monitor.py`)
- Designed and implemented the **`QueryMetrics` dataclass** capturing per-query performance data: query text, pipeline used, start/end time, latency, tool calls list, tool count, retrieval count, risk level, success/failure status, and error message.
- Built the **`MetricsCollector` class** providing aggregate statistics:
  - `total_queries()`, `avg_latency()`, `success_rate()`
  - `pipeline_breakdown()` — count of queries per pipeline
  - `tool_usage_breakdown()` — count of each tool invoked across all queries
  - `avg_latency_by_pipeline()` — per-pipeline average response time

### 2. Live Analytics Dashboard (`app.py` sidebar)
- Integrated `MetricsCollector` with the Streamlit sidebar to display a **real-time analytics panel**:
  - Session-level metrics: total queries, average latency, success rate.
  - Pipeline usage progress bars with per-pipeline breakdown.
  - Per-pipeline average latency comparison badges.
  - Tool call frequency breakdown.

### 3. Per-Response Latency Badges
- Implemented inline **latency tags** (`⏱ X.Xs`) and **risk level badges** under each assistant message.

---

## Lab 8 Contributions

### 1. Domain Task Definition & Model Selection
- Led the team discussion to define the **domain reasoning task**: legal contract risk classification using the CUAD dataset.
- Researched and selected **Llama-3 8B** as the base model for PEFT fine-tuning over alternatives (Mistral-7B, Phi-3).

### 2. PEFT Training Notebook (`LexGuard_PEFT_Training.ipynb`)
- Authored the **Google Colab training notebook** using Unsloth + QLoRA (LoRA rank=16, lora_alpha=16), fine-tuning Llama-3 8B in 4-bit quantization on T4 GPU.
- Pushed the fine-tuned LoRA adapter to HuggingFace Hub as `doandune/LexGuard-llama3-Risk-Adapter`.

### 3. Evaluation Design & Results
- Designed the 10-query evaluation set covering diverse legal reasoning tasks.
- Defined evaluation metrics: answer rate, response time, knowledge source, and reasoning style.

### 4. LocalStore Search Quality (`local_store.py`)
- Extended `LocalStore.search_clauses()` with **phrase-level boosting** and score normalization.

---

## Previous Lab Contributions (Labs 1–7)
- Designed and implemented `LocalStore` class (241 LOC): a three-namespace JSON storage engine (`kv_store_documents.json`, `kv_store_chunks.json`, `kv_store_clause_index.json`).
- Conducted HyperGraphRAG reproduction attempt and documented findings in `RELATED_WORK_REPRO.md`.
- Curated the 23-term legal keyword vocabulary for the inverted clause index.
- Developed `phase_2.ipynb` and `phase_3.ipynb` evaluation notebooks.

---

## Links to Commits
- [Initial commit for Lab 6: LexGuard Agent](https://github.com/Manan151179/BIG_DATA_LAB6/commit/7c4ea6e)
- [Organize artifacts and update smoke tests](https://github.com/Manan151179/BIG_DATA_LAB6/commit/098a9d0)

---

## AI Tools Used
- **Antigravity (Google DeepMind)**: Used to generate the PEFT training notebook structure (Lab 8) and to design the `monitor.py` monitoring module architecture (Lab 9).
- **Gemini API**: Used to generate verbose, explanation-rich outputs for the instruction dataset.

---

## Technical Reflection

Phase 3's evaluation results validated the monitoring infrastructure built in Lab 9: the `MetricsCollector` provided real-time latency comparisons that helped the team quantify the performance difference between BERT extraction and full-document LLM extraction. The `QueryMetrics` dataclass proved flexible enough to accommodate the new extraction approach without any schema changes — the `tool_calls` field simply captures different tool names (`extract_risk_clauses_llm` vs `extract_clause_with_bert`) while the `latency` and `success` fields remain universal. This confirmed that designing metrics around pipeline-agnostic abstractions rather than specific tool implementations was the right architectural choice.
