# Individual Contribution Report — Labs 8 & 9
**Name:** Joe Doan
**Role:** Data Pipeline Engineer & Domain Adaptation Lead

---

## Lab 8 Contributions

### 1. Instruction Dataset Generation (`generate_dataset.py`)
- Designed and implemented the **instruction dataset pipeline** using the Gemini API to generate 50 high-quality training examples from the CUAD contract corpus.
- Each example follows the required Lab 8 format with `instruction`, `input`, and `output` fields, producing verbose, step-by-step legal reasoning outputs suitable for fine-tuning.
- Implemented batch processing with automatic retry logic and rate-limiting to handle the Gemini API quota constraints.
- Saved the final dataset as `instruction_dataset.json` (239 KB, 50 examples).

### 2. Domain-Adapted Agent Pipeline (`adapted_agent.py`)
- Designed and implemented the **full PEFT inference pipeline** that integrates the fine-tuned Llama-3 adapter with the local RAG store:
  - `LLAMA3_PROMPT_TEMPLATE`: Correct Llama-3 special token formatting (`<|begin_of_text|>`, `<|start_header_id|>`, `<|eot_id|>`) for proper instruction following.
  - `query_colab_api()`: HTTP client function that sends formatted prompts to the Colab-hosted FastAPI server via Ngrok tunnel, with robust response parsing to extract only the model's generated answer.
  - `run_adapted_agent()`: Full RAG-to-model-to-risk-assessment pipeline with greeting filtering to prevent unnecessary API calls.
- Debugged the critical **prompt format mismatch bug**: the Llama-3 adapter was receiving Mistral `[INST]` tags, causing an infinite generation loop. Fixed by implementing the correct Llama-3 header token format.
- Implemented robust **response parsing** by identifying that Unsloth's tokenizer renders special tokens as plain text (e.g., `{user_query}assistant\n{answer}`), and splitting on `"assistant\n"` to extract only the generated answer.

### 3. FastAPI Colab Server Configuration
- Configured the **Google Colab inference server** (`GenerateRequest` schema, `generate()` endpoint) to correctly accept and process generation parameters without deadlocking the GPU.
- Diagnosed and fixed the Llama-3 GPU deadlock caused by incorrect input format, implementing `torch.inference_mode()` and explicit `pad_token_id` to stabilize generation.

### 4. Lab 8 Evaluation (`run_evaluation.py`, `EVALUATION.md`)
- Wrote the `run_evaluation.py` script that ran both agents on all 10 queries live and recorded timing and answer-rate metrics.
- Authored `EVALUATION.md` with a full comparison table, summary metrics, and analysis of the baseline vs. adapted system trade-offs.

---

## Previous Lab Contributions (Labs 1–7)
- Built `ingest.py`: complete PDF → Snowflake ingestion pipeline with OCR fallback, dual-write to `LocalStore`, and deterministic UUID chunk IDs.
- Curated and organized the 6 contract PDFs in `./data/` from public SEC/CUAD sources.
- Implemented `get_snowflake_connection()` with auto-provisioning (CREATE IF NOT EXISTS) and MFA support.

---

## Links to Commits
- [Initial commit: environment setup](https://github.com/Manan151179/BIG_DATA_LAB6/commit/b6abf4a)
- [Initial commit for Lab 6: LexGuard Agent](https://github.com/Manan151179/BIG_DATA_LAB6/commit/7c4ea6e)

---

## AI Tools Used
- **Antigravity (Google DeepMind)**: Used to debug the Llama-3 prompt formatting bug, identify the exact Unsloth tokenizer output format via live API testing, and generate the `EVALUATION.md` report.
- **Gemini API**: Used to generate the `instruction_dataset.json` training examples.

---

## Technical Reflection

The most difficult challenge in Lab 8 was debugging an invisible GPU deadlock: the Llama-3 model would accept requests but hang indefinitely, producing no output or error. By writing a minimal test script (`test_colab.py`) that sent a single `repr()`-printed request and reading the raw bytes of the tokenizer's output, I discovered that Unsloth renders Llama-3's special tokens (`<|start_header_id|>assistant<|end_header_id|>`) as plain text without surrounding newlines: `...{query}assistant\n{answer}`. This required a very precise string split rather than a regex, and taught me that low-level tokenizer behavior often contradicts higher-level documentation.

---

## Lab 9 Contributions

### 1. Structured Execution Traces (`agent.py`, `adapted_agent.py`)
- Redesigned both agent functions to return **structured dict responses** instead of plain strings, containing: `response`, `trace`, `tool_calls`, `retrieval_count`, `risk_level`, and `success`.
- Implemented **per-step timing** in the baseline agent: each Gemini tool call is wrapped with `time.time()` measurements, capturing tool name, arguments, result preview, and elapsed time.
- Implemented **three-phase timing** in the adapted agent: RAG retrieval time, Colab model inference time, and risk calculation time are each individually measured and recorded in the trace.
- The trace data feeds directly into the UI debug panels and the `MetricsCollector` analytics system.

### 2. Debug Logging Integration
- Added structured trace entries for every execution step: `start`, `tool_call`, `model_inference`, `response`, `error`, `greeting_filter`, `no_results`, and `timeout`.
- Each trace entry includes a `result_preview` (first 150 chars) so the UI can display tool outputs without overwhelming the interface.

### 3. Development Report (`LAB9_REPORT.md`)
- Authored the 1–2 page group development report covering all 4 enhancement areas, deployment method, and how the system extends Phase-2.

---

## AI Tools Used (Lab 9)
- **Antigravity (Google DeepMind)**: Used to implement structured execution traces in both agents, design the trace data format, and author `LAB9_REPORT.md`.

