# Team Contribution Report — Labs 8 & 9
**Project:** LexGuard — Domain-Adapted Legal Compliance Auditor
**Lab 8:** Fine-Tuning and Domain Adaptation for GenAI Systems
**Lab 9:** Application and Deployment Enhancement
**Deadline:** March 20, 2026

---

## Lab 9 — Team Contribution Table

| Team Member | Role | Lab 9 Contributions | % |
|---|---|---|---|
| **Joe Doan** | Data Pipeline & Adaptation Lead | Structured execution traces in `agent.py` and `adapted_agent.py`, timed tool calls, trace-based debug logging, `LAB9_REPORT.md` | 30% |
| **Manan Koradiya** | Agent Architect & Integrator | Complete `app.py` UI redesign with premium dark theme, glassmorphism CSS, chat interface, reasoning panels, query history sidebar, error handling | 25% |
| **Aditya Naredla** | Storage & Evaluation Engineer | `monitor.py` module (`QueryMetrics` + `MetricsCollector`), live analytics dashboard in sidebar, per-pipeline latency comparison | 25% |
| **Ruixuan Hou** | Reproducibility Lead | `requirements.txt`, `.streamlit/config.toml`, `Dockerfile`, deployment configuration, system status panel | 20% |
| **Total** | | | **100%** |

---

## Lab 9 — System Architecture

```
User Query (Streamlit UI — Premium Dark Theme)
        ↓
  [Pipeline Selector]
  /             \
Baseline       Adapted
(agent.py)  (adapted_agent.py)
  |                 |
Snowflake      LocalStore
RAG Retrieval  RAG Retrieval
  |                 |
Gemini 2.5    Llama-3 8B
Flash API     QLoRA PEFT
              (Colab + Ngrok)
  \                 /
   Risk Assessment
   (calculate_risk_level)
        ↓
  Structured Execution Trace
  (timed tool calls + reasoning steps)
        ↓
  MetricsCollector (monitor.py)
        ↓
  Streamlit Response
  + Debug Panel + Analytics Dashboard
```

---

## Lab 9 — Enhancement Areas Covered

| Area | Enhancement | Files Modified/Created |
|---|---|---|
| A. UI & Workflow | Premium dark theme, glassmorphism, reasoning panels, query history | `app.py` |
| B. Monitoring | QueryMetrics, session analytics, per-pipeline comparison | `monitor.py`, `app.py` |
| C. Logging | Structured traces with per-step timing in both agents | `agent.py`, `adapted_agent.py`, `app.py` |
| D. Deployment | Docker, Streamlit config, requirements, error handling | `Dockerfile`, `.streamlit/config.toml`, `requirements.txt`, `app.py` |

---

## Lab 8 — Team Contribution Table

| Team Member | Role | Lab 8 Contributions | % |
|---|---|---|---|
| **Joe Doan** | Data Pipeline & Adaptation Lead | Instruction dataset generation (`generate_dataset.py`), `adapted_agent.py` full pipeline, Colab FastAPI server debugging, prompt format fix, response parsing, `EVALUATION.md` | 30% |
| **Manan Koradiya** | Agent Architect & Integrator | Streamlit baseline vs. adapted toggle (`app.py`), RAG fallback enhancement (`tools.py`), end-to-end system integration | 25% |
| **Aditya Naredla** | Storage & Evaluation Engineer | Domain task definition, model selection (Llama-3), PEFT training notebook (`LexGuard_PEFT_Training.ipynb`), HuggingFace Hub adapter upload, evaluation design | 25% |
| **Ruixuan Hou** | Reproducibility Lead | `reproduce.sh` Lab 8 updates, new smoke tests for adapted pipeline, `REPRO_AUDIT.md` non-determinism documentation, `RUN.md` setup instructions | 20% |
| **Total** | | | **100%** |

---

## Deliverables Summary

| Deliverable | File | Status |
|---|---|---|
| Premium Streamlit UI | `app.py` | ✅ Dark theme + glassmorphism |
| Monitoring Module | `monitor.py` | ✅ QueryMetrics + Analytics |
| Structured Traces | `agent.py`, `adapted_agent.py` | ✅ Timed tool calls |
| Deployment Config | `Dockerfile`, `.streamlit/config.toml` | ✅ Docker + Theme |
| Dependencies | `requirements.txt` | ✅ Pinned versions |
| Development Report | `LAB9_REPORT.md` | ✅ 1-2 pages |
| Individual Reports | `CONTRIBUTION_*.md` | ✅ All 4 members |