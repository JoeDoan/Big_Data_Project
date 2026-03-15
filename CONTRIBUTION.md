# Team Contribution Report — Lab 8
**Project:** LexGuard — Domain-Adapted Legal Compliance Auditor
**Lab:** Lab 8 – Fine-Tuning and Domain Adaptation for GenAI Systems
**Deadline:** March 16, 2026

---

## Team Contribution Table

| Team Member | Role | Lab 8 Contributions | % |
|---|---|---|---|
| **Joe Doan** | Data Pipeline & Adaptation Lead | Instruction dataset generation (`generate_dataset.py`), `adapted_agent.py` full pipeline, Colab FastAPI server debugging, prompt format fix, response parsing, `EVALUATION.md` | 30% |
| **Manan Koradiya** | Agent Architect & Integrator | Streamlit baseline vs. adapted toggle (`app.py`), RAG fallback enhancement (`tools.py`), end-to-end system integration | 25% |
| **Aditya Naredla** | Storage & Evaluation Engineer | Domain task definition, model selection (Llama-3), PEFT training notebook (`LexGuard_PEFT_Training.ipynb`), HuggingFace Hub adapter upload, evaluation design | 25% |
| **Ruixuan Hou** | Reproducibility Lead | `reproduce.sh` Lab 8 updates, new smoke tests for adapted pipeline, `REPRO_AUDIT.md` non-determinism documentation, `RUN.md` setup instructions | 20% |
| **Total** | | | **100%** |

---

## System Architecture (Lab 8)

```
User Query (Streamlit UI)
        ↓
  [Agent Toggle]
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
  Streamlit Response
```

---

## Deliverables Summary

| Deliverable | File | Status |
|---|---|---|
| Instruction Dataset | `instruction_dataset.json` | ✅ 50 examples |
| Training Notebook | `LexGuard_PEFT_Training.ipynb` | ✅ QLoRA on T4 |
| Adapted Model | `doandune/LexGuard-llama3-Risk-Adapter` | ✅ HuggingFace Hub |
| Integrated Pipeline | `adapted_agent.py` | ✅ RAG + PEFT |
| Demo UI | `app.py` (baseline vs. adapted toggle) | ✅ Streamlit |
| Evaluation Report | `EVALUATION.md` | ✅ 10 queries |
| Individual Reports | `CONTRIBUTION_*.md` | ✅ All 4 members |