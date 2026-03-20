# Individual Contribution Report — Labs 8 & 9
**Name:** Aditya Naredla
**Role:** Storage Architect & Evaluation Engineer

---

## Lab 8 Contributions

### 1. Domain Task Definition & Model Selection
- Led the team discussion to define the **domain reasoning task**: legal contract risk classification using the CUAD dataset, selecting clause-level risk analysis (High/Medium/Low) as the specific instruction tuning objective.
- Researched and selected **Llama-3 8B** as the base model for PEFT fine-tuning over alternatives (Mistral-7B, Phi-3), based on its improved instruction-following capability and legal reasoning benchmark performance.
- Justified the model choice in the group report: Llama-3's chat template uses distinct role headers (`<|start_header_id|>system/user/assistant<|end_header_id|>`) that align well with structured legal audit prompts.

### 2. PEFT Training Notebook (`LexGuard_PEFT_Training.ipynb`)
- Authored the **Google Colab training notebook** using Unsloth + QLoRA (LoRA rank=16, lora_alpha=16), fine-tuning Llama-3 8B in 4-bit quantization on T4 GPU.
- Configured the training loop: `SFTTrainer` with 60 steps, batch size=2, gradient accumulation=4, learning rate=2e-4, AdamW 8-bit optimizer.
- Pushed the fine-tuned LoRA adapter to HuggingFace Hub as `doandune/LexGuard-llama3-Risk-Adapter` for team-wide access.
- Verified training convergence by monitoring the SFT loss curve and confirming the adapter learned legal risk classification format from the 50-example dataset.

### 3. Evaluation Design & Results (`EVALUATION.md`, `eval_results.json`)
- Designed the 10-query evaluation set covering diverse legal reasoning tasks: change of control, liability caps, termination conditions, party identification, confidentiality, indemnification, governing law, assignment restrictions, insurance, and payment obligations.
- Defined evaluation metrics: answer rate (binary), response time, knowledge source, and reasoning style.
- Produced `EVALUATION.md` documenting the full comparison table and analysis of trade-offs between baseline and adapted systems.

### 4. LocalStore Search Quality (`local_store.py`)
- Extended the `LocalStore.search_clauses()` method with **phrase-level boosting**: clauses containing multi-word legal phrases (e.g., "change of control", "indemnification") score higher than clauses with isolated keyword matches, improving retrieval precision for the adapted agent.
- Added score normalization to ensure results are consistent regardless of clause length.

---

## Lab 9 Contributions

### 1. Monitoring Module (`monitor.py`)
- Designed and implemented the **`QueryMetrics` dataclass** capturing per-query performance data: query text, pipeline used, start/end time, latency, tool calls list, tool count, retrieval count, risk level, success/failure status, and error message.
- Built the **`MetricsCollector` class** that accumulates metrics across a Streamlit session, providing aggregate statistics:
  - `total_queries()`, `avg_latency()`, `success_rate()`
  - `pipeline_breakdown()` — count of queries per pipeline
  - `tool_usage_breakdown()` — count of each tool invoked across all queries
  - `avg_latency_by_pipeline()` — per-pipeline average response time for comparison

### 2. Live Analytics Dashboard (`app.py` sidebar)
- Integrated `MetricsCollector` with the Streamlit sidebar to display a **real-time analytics panel** that appears after the first query:
  - Session-level metrics: total queries, average latency, success rate
  - Pipeline usage progress bars with per-pipeline breakdown
  - Per-pipeline average latency comparison badges
  - Tool call frequency breakdown showing which tools are most used

### 3. Per-Response Latency Badges
- Implemented inline **latency tags** (`⏱ X.Xs`) and **risk level badges** displayed under each assistant message, providing immediate feedback on system performance without expanding the debug panel.

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
- **Antigravity (Google DeepMind)**: Used to generate the PEFT training notebook structure (Lab 8) and to design the `monitor.py` monitoring module architecture and analytics dashboard integration (Lab 9).
- **Gemini API**: Used to generate verbose, explanation-rich outputs for the instruction dataset.

---

## Technical Reflection

Lab 9's monitoring challenge was deciding what to track without adding overhead. The `QueryMetrics` dataclass uses Python's `time.time()` for microsecond-precision latency measurement — cheap enough to run on every query. The key design decision was storing metrics in `st.session_state` rather than writing to disk, which makes the analytics dashboard zero-latency (no file I/O) and ensures metrics don't persist across sessions (avoiding stale data from different deployment contexts). The per-pipeline comparison feature directly extends the Lab 8 evaluation work: instead of running offline batch evaluations, users can now see baseline vs. adapted performance differences live in the sidebar.
