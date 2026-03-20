# Individual Contribution Report — Labs 8 & 9
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

## Lab 9 Contributions

### 1. Complete Streamlit UI Redesign (`app.py`)
- Rewrote `app.py` from scratch with a **premium dark theme** featuring:
  - Custom CSS with glassmorphism cards, gradient borders, and hover animations
  - Inter font from Google Fonts for modern typography
  - Animated gradient header ("⚖️ LexGuard") using CSS `background-clip: text`
  - Color-coded risk badges: 🟢 Low (green), 🟡 Medium (amber), 🔴 High (red)
  - Per-response latency tags showing execution time inline

### 2. Query History Sidebar
- Implemented a **clickable query history panel** in the sidebar that shows the last 10 queries with their latency and risk level, giving users a quick overview of their session.

### 3. Execution Trace & Reasoning Panels
- Built **collapsible "🔍 Execution Trace & Debug Log" expanders** under each assistant response, visualizing the step-by-step reasoning process:
  - `📝 Query Received` → `🛠️ Tool Call` → `🤖 Model Inference` → `✅ Response Generated`
  - Each step shows timing and result previews with styled trace blocks.

### 4. Error Handling & System Status (Area D)
- Wrapped all agent calls in `try/except` blocks with user-friendly error messages instead of raw stack traces.
- Added a **System Status panel** in the sidebar showing live connectivity indicators for Gemini API, Snowflake DB, and Colab PEFT Server.
- Implemented wide layout mode for better content utilization.

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
- **Antigravity (Google DeepMind)**: Used to implement the RAG fallback logic in `tools.py` (Lab 8) and to build the premium dark-theme Streamlit UI with glassmorphism CSS and execution trace panels (Lab 9).

---

## Technical Reflection

In Lab 9, the core UI challenge was making the execution traces readable without cluttering the chat interface. The solution was Streamlit's `st.expander()`, which hides the debug panel by default but makes it instantly accessible. The CSS glassmorphism effect (`backdrop-filter: blur(12px)` + semi-transparent backgrounds) creates visual depth that separates the sidebar analytics from the main chat area, making the interface feel professional rather than overwhelming despite showing a lot of real-time data.
