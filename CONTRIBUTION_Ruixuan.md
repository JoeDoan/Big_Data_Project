# Individual Contribution Report — Labs 1–9 (Through Phase 3)
**Name:** Ruixuan Hou
**Role:** Reproducibility & Testing Lead

---

## Phase 3 Contributions

### 1. Dependency Management Updates (`requirements.txt`)
- Updated `requirements.txt` to include all Phase 3 dependencies, maintaining **pinned minimum versions** for reproducible installs across team members and deployment environments.
- Ensured compatibility across macOS ARM, Linux x86, and Docker deployment targets.

### 2. Docker Configuration Maintenance (`Dockerfile`)
- Maintained the **production Dockerfile** to support the Phase 3 features (chat history, theme toggle) without requiring container rebuild for configuration changes.
- Environment variables for Snowflake chat persistence (`SNOW_USER`, `SNOW_PASS`, `SNOW_ACCOUNT`, etc.) are injected via `--env-file .env` at runtime.

### 3. Streamlit Theme Configuration (`.streamlit/config.toml`)
- Updated `.streamlit/config.toml` to coordinate with the new dark/light theme toggle system, ensuring Streamlit's native widgets match the custom CSS in both modes.

### 4. Reproducibility Documentation
- Updated `REPRO_AUDIT.md` to document the Phase 3 pipeline changes:
  - Noted that the BERT model (`doandune/LexGuard-CUAD-BERT`) is deprecated from production but remains available on HuggingFace Hub for reference.
  - Documented the new Snowflake tables (`CHAT_SESSIONS`, `CHAT_MESSAGES`) and their auto-provisioning via `chat_history.init_tables()`.
  - Updated the non-determinism boundary: Gemini API responses may vary between runs due to model versioning on Google's side.

---

## Lab 9 Contributions

### 1. Dependency Management (`requirements.txt`)
- Rewrote `requirements.txt` with **pinned minimum versions** for all production dependencies: `streamlit>=1.30.0`, `google-genai>=1.0.0`, `snowflake-connector-python>=3.6.0`, etc.

### 2. Streamlit Configuration (`.streamlit/config.toml`)
- Created `.streamlit/config.toml` with a **custom dark purple theme** (`primaryColor=#7C3AED`, `backgroundColor=#0F172A`) matching the glassmorphism CSS in `app.py`.
- Configured headless server mode for deployment and disabled usage stats collection.

### 3. Docker Deployment (`Dockerfile`)
- Authored a **production Dockerfile** using `python:3.12-slim` base image:
  - Installs system build dependencies, then Python packages from `requirements.txt`.
  - Copies application code, exposes port 8501.
  - Includes a `HEALTHCHECK` command that verifies the Streamlit server is responsive.

### 4. System Status Panel
- Contributed the **system status indicator panel** in the sidebar checking environment variables for connectivity status (🟢 Online / 🔴 Offline).

---

## Lab 8 Contributions

### 1. Reproducibility of PEFT Pipeline
- Updated `REPRO_AUDIT.md` to document the Lab 8 domain adaptation pipeline reproducibility:
  - Documented Colab environment requirements and non-determinism boundaries.
  - Noted the fine-tuned LoRA adapter is pinned on HuggingFace Hub at `doandune/LexGuard-llama3-Risk-Adapter`.

### 2. Updated `reproduce.sh` for Lab 8
- Extended `reproduce.sh` to include Lab 8 dependencies and evaluation runner.
- Added validation step for Ngrok API URL before attempting inference.

### 3. Smoke Test Extensions (`tests/test_smoke.py`)
- Added smoke tests for the adapted pipeline:
  - `test_instruction_dataset_format`, `test_adapted_agent_greeting_filter`, `test_llama3_prompt_template`.

### 4. Environment & Dependency Documentation
- Updated `requirements.txt` with all Lab 8 dependencies.
- Updated `RUN.md` with step-by-step instructions for the adapted agent pipeline.

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
- **Antigravity (Google DeepMind)**: Used to design Lab 8 smoke test cases, create the Dockerfile, Streamlit theme configuration, and dependency management files.

---

## Technical Reflection

Phase 3's shift from BERT + chunking to full-document LLM extraction had an interesting reproducibility implication: the system became simultaneously less deterministic (Gemini API responses can vary between calls) and more reliable (higher accuracy means fewer false negatives to debug). The trade-off is documented in `REPRO_AUDIT.md` — we accept API-level non-determinism because the accuracy improvement from 53.8% to 86.3% is worth more than byte-identical reproducibility of incorrect results. The Docker container now serves as the "exact reproduction" path when needed, with all dependencies frozen at build time.
