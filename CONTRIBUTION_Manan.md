# Individual Contribution Report — Lab 8
**Name:** Manan Koradiya
**Role:** Agent Architect & System Integrator

---

## Lab 8 Contributions

### 1. Updated Streamlit UI with Baseline vs. Adapted Comparison (`app.py`)
- Extended the Streamlit chat interface with a **model selection toggle** in the sidebar, allowing users to switch between the Baseline (Gemini) and Adapted (Llama-3 PEFT) agents in real time.
- Implemented conditional routing: when the user selects "Adapted Model (Lab 8)", the app calls `run_adapted_agent()` from `adapted_agent.py`; otherwise it calls `run_lexguard_agent()` from `agent.py`.
- Ensured the UI shows the correct agent label ("🧠 LexGuard (Baseline)" vs. "🔬 LexGuard (Adapted)") in the chat response, making model comparison visually clear for the demo.

### 2. RAG Retrieval Fallback Enhancement (`tools.py`)
- Enhanced `retrieve_local_clauses()` in `tools.py` with a **multi-tier fallback** strategy:
  - Tier 1: Inverted keyword index lookup (exact-match, deterministic).
  - Tier 2: Full-text search across all chunks with stop-word filtering and phrase boosting for legal terms (e.g., "change of control", "merger", "parties").
  - Returns formatted citation strings with `[Source: filename]` headers for each matched chunk.
- Implemented stop-word filtering to prevent common words like "the", "is", "what" from polluting the search query and matching irrelevant chunks.

### 3. System Architecture Integration
- Connected all Lab 8 components end-to-end: `generate_dataset.py` → `instruction_dataset.json` → Colab PEFT training → HuggingFace Hub → `adapted_agent.py` → Streamlit UI.
- Ensured backward compatibility: the Lab 8 adapted pipeline works alongside the original Gemini baseline without disrupting existing functionality.

---

## Previous Lab Contributions (Labs 1–7)
- Designed and implemented the agentic reasoning loop in `agent.py` using the Gemini 2.5 Flash SDK.
- Authored the system prompt defining LexGuard's "Recall-Then-Reason" pipeline with max-steps guard and tool dispatch mechanism.
- Wrapped Snowflake queries and risk-assessment logic into LLM-callable tools in `tools.py`.
- Built the Streamlit chat UI with MFA passthrough for Snowflake authentication.

---

## Links to Commits
- [Initial commit for Lab 6: LexGuard Agent](https://github.com/Manan151179/BIG_DATA_LAB6/commit/7c4ea6e)
- [Organize artifacts and update smoke tests](https://github.com/Manan151179/BIG_DATA_LAB6/commit/098a9d0)

---

## AI Tools Used
- **Antigravity (Google DeepMind)**: Used to implement the RAG fallback logic in `tools.py` and to debug the Streamlit toggle integration between both agents.

---

## Technical Reflection

The key challenge in Lab 8 integration was maintaining backward compatibility between the original Gemini-based agent and the new adapted Llama-3 pipeline. Both agents share the same `tools.py` infrastructure, but use different retrieval backends (Snowflake vs. local JSON store). The main design decision was to make `retrieve_local_clauses()` and `retrieve_contract_clauses()` return identically-formatted strings, so that the downstream risk assessment and UI display code could remain unchanged. The multi-tier RAG fallback in `tools.py` was critical for the adapted agent's answer quality — without it, many legal queries would return zero results from the keyword index, since the index only covers 20 pre-defined legal terms.
