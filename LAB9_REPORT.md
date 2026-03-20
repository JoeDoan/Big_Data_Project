# Lab 9: Development Report — LexGuard

## 1. Application Improvements

### A. UI & Workflow Redesign
The Streamlit application was completely redesigned with a premium dark theme featuring:
- **Glassmorphism cards** with gradient borders and hover animations
- **Custom CSS** with Inter font, animated gradient header, and color-coded risk badges (🟢 Low / 🟡 Medium / 🔴 High)
- **Wide layout** with a restructured sidebar containing pipeline selector, system status indicators (Gemini API, Snowflake DB, Colab PEFT), and clickable query history
- **Execution Trace panels** — each response includes a collapsible "🔍 Debug Log" showing the full step-by-step reasoning: Query → Tool Calls → Retrieved Docs → Response, with per-step timing

### B. Monitoring & Evaluation
A new `monitor.py` module was added with:
- **`QueryMetrics` dataclass** capturing per-query: latency, tool calls, retrieval count, risk level, success/failure
- **`MetricsCollector`** accumulating session metrics with aggregate stats: total queries, average latency, success rate, tool usage breakdown, per-pipeline latency comparison
- **Live analytics dashboard** in the sidebar that updates after each query, showing real-time performance data

### C. Logging & Debugging
Both agent pipelines (`agent.py`, `adapted_agent.py`) were modified to return **structured execution traces** alongside responses:
- Each tool call is individually timed and logged with arguments and result previews
- The baseline agent logs Gemini API tool routing steps
- The adapted agent logs RAG retrieval time, Colab model inference time, and risk calculation time
- Traces are displayed in the UI via expandable panels under each assistant message

## 2. Deployment Method

### Streamlit Configuration
- `.streamlit/config.toml` — custom dark purple theme and headless server config

### Docker (Option 5)
- `Dockerfile` — Python 3.12-slim image, installs dependencies, exposes port 8501, includes healthcheck
- `requirements.txt` — pinned dependency list for reproducible installs

Build and run:
```bash
docker build -t lexguard .
docker run -p 8501:8501 --env-file .env lexguard
```

### Error Handling
- All agent calls wrapped in try/except with user-friendly error messages
- Graceful degradation when services are unavailable

## 3. How This Extends Phase-2

The Phase-2 prototype provided the core RAG pipeline and Streamlit interface. Lab 9 enhances it into a **near-production system** by adding:

| Phase-2 | Lab 9 Enhancement |
|---|---|
| Basic chat UI | Premium glassmorphism dark theme |
| Print-based debugging | Structured execution traces in UI |
| No performance tracking | Real-time analytics dashboard |
| Manual deployment | Docker + Streamlit config |
| No error handling | Graceful error handling throughout |

## 4. Tools Used
- **Antigravity IDE** — AI-assisted development for code generation, debugging, and UI styling
- **GitHub** — version control and collaboration.
