# Individual Contribution Report — Labs 8 & 9
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

## Lab 9 Contributions

### 1. Dependency Management (`requirements.txt`)
- Rewrote `requirements.txt` with **pinned minimum versions** for all production dependencies: `streamlit>=1.30.0`, `google-genai>=1.0.0`, `snowflake-connector-python>=3.6.0`, etc.
- Organized dependencies into a clean list ensuring reproducible installs across team members and deployment environments.

### 2. Streamlit Configuration (`.streamlit/config.toml`)
- Created `.streamlit/config.toml` with a **custom dark purple theme** (`primaryColor=#7C3AED`, `backgroundColor=#0F172A`, `secondaryBackgroundColor=#1E293B`) that matches the glassmorphism CSS in `app.py`.
- Configured headless server mode for deployment and disabled usage stats collection.

### 3. Docker Deployment (`Dockerfile`)
- Authored a **production Dockerfile** using `python:3.12-slim` base image:
  - Installs system build dependencies, then Python packages from `requirements.txt`.
  - Copies application code, exposes port 8501.
  - Includes a `HEALTHCHECK` command that verifies the Streamlit server is responsive.
  - Configurable via `--env-file .env` at runtime for API key injection.
- Documented build/run commands in `LAB9_REPORT.md`.

### 4. System Status Panel
- Contributed the **system status indicator panel** in the sidebar that checks environment variables to determine connectivity status:
  - 🟢 Online: Gemini API key and Snowflake credentials detected
  - 🟡 Unknown: Colab URL set but connectivity unverified
  - 🔴 Offline: Missing credentials

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
- **Antigravity (Google DeepMind)**: Used to design Lab 8 smoke test cases (Lab 8), and to create the Dockerfile, Streamlit theme configuration, and dependency management files (Lab 9).

---

## Technical Reflection

Lab 9's deployment work revealed an important tension between reproducibility and flexibility. The `requirements.txt` uses `>=` version pins rather than `==` exact pins — this is intentional. Exact pins would guarantee byte-identical installs but would break on different Python versions or platforms (e.g., `snowflake-connector-python` has different wheels for macOS ARM vs Linux x86). The `>=` approach ensures the code works on the widest range of deployment targets (Streamlit Cloud, Docker, Colab, local Mac) while the Dockerfile provides an exact-reproduction path when needed. The `.streamlit/config.toml` theme coordination with the CSS in `app.py` was also non-trivial — Streamlit applies its theme to native widgets (radio buttons, metrics, expanders) but not to custom HTML, so both systems need to use the same color palette to avoid visual mismatches.
