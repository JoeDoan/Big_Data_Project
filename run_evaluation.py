#!/usr/bin/env python3
"""
Lab 8 Evaluation Script
Runs 10 legal queries through both Baseline and Adapted agents and records results.
"""
import os
import sys
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# Import both agents
from agent import run_lexguard_agent
from adapted_agent import run_adapted_agent

EVAL_QUERIES = [
    "What happens in the event of a change of control or merger?",
    "Is there any uncapped liability mentioned in the contracts?",
    "What are the termination conditions in the agreements?",
    "Who are the parties involved in the contracts?",
    "What are the confidentiality obligations?",
    "What indemnification obligations does Bachem have?",
    "What is the governing law for dispute resolution?",
    "Are there any assignment restrictions in the agreements?",
    "What insurance requirements are mentioned?",
    "What are the payment or royalty obligations?"
]

results = []

print("=" * 60)
print("LexGuard Lab 8 Evaluation - Baseline vs Adapted")
print("=" * 60)

for i, query in enumerate(EVAL_QUERIES, 1):
    print(f"\n[{i}/10] Query: {query}")
    result = {"query": query}

    # --- Baseline Agent ---
    try:
        print("  → Running baseline agent (Gemini + Snowflake)...")
        t0 = time.time()
        baseline_res = run_lexguard_agent(query)
        result["baseline_time"] = round(time.time() - t0, 1)
        result["baseline_response"] = baseline_res[:300] if baseline_res else "No response"
        result["baseline_answered"] = "No evidence" not in baseline_res and len(baseline_res) > 50
    except Exception as e:
        result["baseline_time"] = 0
        result["baseline_response"] = f"Error: {str(e)[:100]}"
        result["baseline_answered"] = False

    # --- Adapted Agent ---
    try:
        print("  → Running adapted agent (Llama-3 PEFT + Local RAG)...")
        t0 = time.time()
        adapted_res = run_adapted_agent(query)
        result["adapted_time"] = round(time.time() - t0, 1)
        result["adapted_response"] = adapted_res[:300] if adapted_res else "No response"
        result["adapted_answered"] = "Could not find" not in adapted_res and "Error" not in adapted_res and len(adapted_res) > 50
    except Exception as e:
        result["adapted_time"] = 0
        result["adapted_response"] = f"Error: {str(e)[:100]}"
        result["adapted_answered"] = False

    results.append(result)
    print(f"  ✓ Done (baseline: {result['baseline_time']}s, adapted: {result['adapted_time']}s)")

# Save results
with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ Evaluation complete! Results saved to eval_results.json")
