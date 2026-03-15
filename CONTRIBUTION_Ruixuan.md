# Individual Contribution Report — Lab 8
**Name:** Ruixuan Hou
**Role:** Reproducibility & Testing Lead

---

## Lab 8 Contributions

### 1. Reproducibility of PEFT Pipeline
- Updated `REPRO_AUDIT.md` to document the Lab 8 domain adaptation pipeline reproducibility:
  - Documented the Colab environment: Unsloth + QLoRA dependencies, T4 GPU requirements, and Python version constraints.
  - Noted the **non-determinism boundary**: model training steps are deterministic given `seed=3407`; however, GPU floating-point operations (cuDNN) may produce ±0.1% loss variation across hardware.
  - Documented that the fine-tuned LoRA adapter is pinned on HuggingFace Hub at `doandune/LexGuard-llama3-Risk-Adapter` to ensure reproducible inference.

### 2. Updated `reproduce.sh` for Lab 8
- Extended `reproduce.sh` to include Lab 8 dependencies: `python-dotenv`, `requests`, and the evaluation runner.
- Added a `run_evaluation.py` execution step that automatically re-runs the 10-query evaluation and saves results to `eval_results.json` and `EVALUATION.md`.
- Added a validation step that checks the Ngrok API URL is set in `.env` before attempting inference.

### 3. Smoke Test Extensions (`tests/test_smoke.py`)
- Added smoke tests for the Lab 8 adapted pipeline:
  - `test_instruction_dataset_format` — validates `instruction_dataset.json` has 20+ examples and each entry contains `instruction`, `input`, and `output` keys.
  - `test_adapted_agent_greeting_filter` — verifies that the greeting filter in `run_adapted_agent()` short-circuits correctly without hitting the RAG pipeline.
  - `test_llama3_prompt_template` — confirms the `LLAMA3_PROMPT_TEMPLATE` contains required Llama-3 special tokens (`<|begin_of_text|>`, `<|start_header_id|>`, `<|eot_id|>`).

### 4. Environment & Dependency Documentation
- Updated `requirements.txt` with all Lab 8 dependencies (python-docx, requests, etc.).
- Updated `RUN.md` with step-by-step instructions for setting up the adapted agent pipeline:
  - How to get the Ngrok URL from Colab.
  - How to set `COLAB_API_URL` in `.env`.
  - How to switch between agents in the Streamlit UI.

---

## Previous Lab Contributions (Labs 1–7)
- Designed and implemented the centralized configuration module (`config.py`) with seeded RNG, deterministic UUIDs, and HYPERPARAMS dictionary.
- Built structured logging framework (`lexguard_logger.py`) with dual output and run manifest generation.
- Authored all 11 smoke tests in `tests/test_smoke.py` covering the full pipeline without external API dependencies.
- Wrote `reproduce.sh` and `REPRO_AUDIT.md` for one-command pipeline reproducibility.

---

## Links to Commits
- [Initial commit: environment setup](https://github.com/Manan151179/BIG_DATA_LAB6/commit/b6abf4a)
- [Organize artifacts and update smoke tests](https://github.com/Manan151179/BIG_DATA_LAB6/commit/098a9d0)

---

## AI Tools Used
- **Antigravity (Google DeepMind)**: Used to design the Lab 8 smoke test cases and identify which non-determinism boundaries needed explicit documentation in `REPRO_AUDIT.md`.

---

## Technical Reflection

Lab 8 introduced a fundamentally new category of non-determinism: **model training non-determinism**. In previous labs, our reproducibility guarantees covered data processing (ingestion, chunking, UUID generation) and retrieval (keyword search, JSON serialization). But fine-tuning a neural network on a GPU introduces floating-point precision non-determinism that cannot be fully controlled — even with `seed=3407` and cuDNN deterministic mode, different GPU hardware (A100 vs T4) may produce slightly different loss curves and therefore slightly different adapter weights. My approach was to accept this as a documented limitation: in `REPRO_AUDIT.md`, I formally defined the reproducibility guarantee as "identical results given the same HuggingFace Hub adapter checkpoint" rather than "identical results given the same training run," which is a more honest and useful contract for a production system.
