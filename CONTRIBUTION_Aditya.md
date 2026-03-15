# Individual Contribution Report — Lab 8
**Name:** Aditya Naredla
**Role:** Storage Architect & Evaluation Engineer

---

## Lab 8 Contributions

### 1. Domain Task Definition & Model Selection
- Led the team discussion to define the **domain reasoning task**: legal contract risk classification using the CUAD dataset, selecting clause-level risk analysis (High/Medium/Low) as the specific instruction tuning objective.
- Researched and selected **Llama-3 8B** as the base model for PEFT fine-tuning over alternatives (Mistral-7B, Phi-3), based on its improved instruction-following capability and legal reasoning benchmark performance.
- Justified the model choice in the group report: Llama-3's chat template uses distinct role headers (`<|start_header_id|>system/user/assistant<|end_header_id|>`) that align well with structured legal audit prompts.

### 2. PEFT Training Notebook (`LexGuard_PEFT_Training.ipynb`)
- Authored the **Google Colab training notebook** using Unsloth + QLoRA (LoRA rank=16, lora_alpha=16), fine-tuning Llama-3 8B in 4-bit quantization on T4 GPU.
- Configured the training loop: `SFTTrainer` with 60 steps, batch size=2, gradient accumulation=4, learning rate=2e-4, AdamW 8-bit optimizer.
- Pushed the fine-tuned LoRA adapter to HuggingFace Hub as `doandune/LexGuard-llama3-Risk-Adapter` for team-wide access.
- Verified training convergence by monitoring the SFT loss curve and confirming the adapter learned legal risk classification format from the 50-example dataset.

### 3. Evaluation Design & Results (`EVALUATION.md`, `eval_results.json`)
- Designed the 10-query evaluation set covering diverse legal reasoning tasks: change of control, liability caps, termination conditions, party identification, confidentiality, indemnification, governing law, assignment restrictions, insurance, and payment obligations.
- Defined evaluation metrics: answer rate (binary), response time, knowledge source, and reasoning style.
- Produced `EVALUATION.md` documenting the full comparison table and analysis of trade-offs between baseline and adapted systems.

### 4. LocalStore Search Quality (`local_store.py`)
- Extended the `LocalStore.search_clauses()` method with **phrase-level boosting**: clauses containing multi-word legal phrases (e.g., "change of control", "indemnification") score higher than clauses with isolated keyword matches, improving retrieval precision for the adapted agent.
- Added score normalization to ensure results are consistent regardless of clause length.

---

## Previous Lab Contributions (Labs 1–7)
- Designed and implemented `LocalStore` class (241 LOC): a three-namespace JSON storage engine (`kv_store_documents.json`, `kv_store_chunks.json`, `kv_store_clause_index.json`).
- Conducted HyperGraphRAG reproduction attempt and documented findings in `RELATED_WORK_REPRO.md`.
- Curated the 23-term legal keyword vocabulary for the inverted clause index.
- Developed `phase_2.ipynb` and `phase_3.ipynb` evaluation notebooks.

---

## Links to Commits
- [Initial commit for Lab 6: LexGuard Agent](https://github.com/Manan151179/BIG_DATA_LAB6/commit/7c4ea6e)
- [Organize artifacts and update smoke tests](https://github.com/Manan151179/BIG_DATA_LAB6/commit/098a9d0)

---

## AI Tools Used
- **Antigravity (Google DeepMind)**: Used to generate the PEFT training notebook structure and debug the QLoRA configuration for Llama-3 on Colab T4 GPU.
- **Gemini API**: Used to generate verbose, explanation-rich outputs for the instruction dataset.

---

## Technical Reflection

The biggest insight from Lab 8 was discovering the "hardware ceiling" effect in fine-tuning: with only 50 examples and 60 training steps, the model doesn't truly learn new legal knowledge — it learns the *format* of legal reasoning. The pre-trained Llama-3 base already contains enough legal knowledge from its training corpus; what PEFT adds is the structural habit of answering in a "Summary → Step-by-Step → Citation" format that matches legal audit best practices. This is why both agents achieved 100% answer rate — the adapted model's advantage is not coverage, but structured presentation quality, which is exactly what domain adaptation theory predicts.
