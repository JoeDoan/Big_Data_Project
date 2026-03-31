# Individual Contribution Report — Labs 1–9 (Through Phase 3)
**Name:** Manan Koradiya
**Role:** Agent Architect & System Integrator

---

## Phase 3 Contributions

### 1. Streamlit UI Architecture (`app.py`)
- Maintained and extended the **premium dark theme** Streamlit interface (720+ lines) with glassmorphism cards, gradient borders, and hover animations.
- Integrated the new Phase 3 features into the existing UI framework:
  - Dark/light theme toggle compatibility with all existing CSS components.
  - Chat history sidebar panel with session management controls.
  - Expandable source annotation rendering for risk clause audit results.

### 2. Agent Routing & Integration
- Ensured the **Dual-Path Architecture** in `agent.py` correctly routes user queries:
  - Path A (Surgical Strike): Tool-based extraction for specific clause questions.
  - Path B (General Answer): Full-document LLM pass for broad contract questions.
- Maintained backward compatibility with the adapted pipeline (`adapted_agent.py`) alongside the new full-document extraction flow.

### 3. RAG Fallback Enhancement (`tools.py`)
- Maintained the **multi-tier fallback** strategy in `retrieve_local_clauses()`:
  - Tier 1: Inverted keyword index lookup (exact-match, deterministic).
  - Tier 2: Full-text search with stop-word filtering and phrase boosting for legal terms.
  - Returns formatted citation strings with `[Source: filename]` headers.

---

## Lab 9 Contributions

### 1. Complete Streamlit UI Redesign (`app.py`)
- Rewrote `app.py` from scratch with a **premium dark theme** featuring:
  - Custom CSS with glassmorphism cards, gradient borders, and hover animations.
  - Inter font from Google Fonts for modern typography.
  - Animated gradient header ("⚖️ LexGuard") using CSS `background-clip: text`.
  - Color-coded risk badges: 🟢 Low (green), 🟡 Medium (amber), 🔴 High (red).
  - Per-response latency tags showing execution time inline.

### 2. Query History Sidebar
- Implemented a **clickable query history panel** in the sidebar showing the last 10 queries with latency and risk level.

### 3. Execution Trace & Reasoning Panels
- Built **collapsible "🔍 Execution Trace & Debug Log" expanders** under each assistant response, visualizing the step-by-step reasoning process with timing and result previews.

### 4. Error Handling & System Status
- Wrapped all agent calls in `try/except` blocks with user-friendly error messages.
- Added a **System Status panel** in the sidebar showing live connectivity indicators for Gemini API, Snowflake DB, and Colab PEFT Server.

---

## Lab 8 Contributions

### 1. Updated Streamlit UI with Baseline vs. Adapted Comparison (`app.py`)
- Extended the Streamlit chat interface with a **model selection toggle** allowing users to switch between Baseline (Gemini) and Adapted (Llama-3 PEFT) agents in real time.
- Implemented conditional routing between `run_adapted_agent()` and `run_lexguard_agent()`.

### 2. RAG Retrieval Fallback Enhancement (`tools.py`)
- Enhanced `retrieve_local_clauses()` with multi-tier fallback strategy (keyword index → full-text search with stop-word filtering).

### 3. System Architecture Integration
- Connected all Lab 8 components end-to-end: `generate_dataset.py` → `instruction_dataset.json` → Colab PEFT training → HuggingFace Hub → `adapted_agent.py` → Streamlit UI.

---

## Previous Lab Contributions (Labs 1–7)
- Designed and implemented the agentic reasoning loop in `agent.py` using the Gemini 2.5 Flash SDK.
- Authored the system prompt defining LexGuard's "Recall-Then-Reason" pipeline with max-steps guard and tool dispatch.
- Wrapped Snowflake queries and risk-assessment logic into LLM-callable tools in `tools.py`.
- Built the initial Streamlit chat UI with MFA passthrough for Snowflake authentication.

---

## Links to Commits
- [Initial commit for Lab 6: LexGuard Agent](https://github.com/Manan151179/BIG_DATA_LAB6/commit/7c4ea6e)
- [Organize artifacts and update smoke tests](https://github.com/Manan151179/BIG_DATA_LAB6/commit/098a9d0)

---

## AI Tools Used
- **Antigravity (Google DeepMind)**: Used to implement RAG fallback logic (Lab 8), build the premium dark-theme Streamlit UI with glassmorphism CSS and execution trace panels (Lab 9), and integrate Phase 3 UI features.

---

## Technical Reflection

In Phase 3, the challenge was integrating the new chat persistence and theme toggle features into an already complex 720-line Streamlit application without breaking existing functionality. The CSS variable approach for theming was critical — rather than duplicating styles for dark and light modes, all colors reference CSS custom properties (e.g., `var(--bg-primary)`) that switch values based on the active theme class. This made adding new components theme-aware by default and kept the CSS manageable despite the growing feature set.
