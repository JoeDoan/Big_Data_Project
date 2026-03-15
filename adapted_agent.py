import os
import requests
import json
import logging
from dotenv import load_dotenv

# Import the tools we defined in tools.py
from tools import retrieve_local_clauses, calculate_risk_level

# Configure logging for the Streamlit terminal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API Key from .env
load_dotenv()
COLAB_API_URL = os.getenv("COLAB_API_URL")

# 1. The System Prompt (Adapted for Llama-3 Instruction Format)
LLAMA3_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are LexGuard, a highly analytical Neuro-Symbolic Compliance Auditor. Your job is to audit Residential Lease Agreements or business contracts.

Follow this strict 'Recall-Then-Reason' pipeline:
1. Carefully read the Extracted Contract Clauses provided below.
2. Answer the User Query based ONLY on those clauses.
3. If the provided clauses do not contain the answer or are irrelevant, explicitly state "Information not found in the provided clauses." and DO NOT attempt to guess or hallucinate.
4. If you have the answer, you MUST write a short summary paragraph first, and then quote ONLY the specific short sentences that support your answer.
5. Provide a detailed, step-by-step explanation of your reasoning.<|eot_id|><|start_header_id|>user<|end_header_id|>

Extracted Contract Clauses:
{context}

User Query:
{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


def query_colab_api(formatted_prompt: str) -> str:
    """Sends the prompt to the Google Colab Ngrok API tunnel."""
    if not COLAB_API_URL or "<your-ngrok-id>" in COLAB_API_URL:
        return "⚠️ Error: Colab API URL (COLAB_API_URL) is missing or not updated in the .env file!"
        
    payload = {
        "text": formatted_prompt,
        "max_tokens": 600,
        "temperature": 0.3, # Increased slightly to work better with sampling
        "top_p": 0.9,       # Nucleus sampling to prevent repetitive loops naturally
        "do_sample": True,  # Turn on sampling
        "repetition_penalty": 1.05 # Lowered! 1.3 is far too toxic for Mistral and causes gibberish
    }
    
    try:
        response = requests.post(COLAB_API_URL, json=payload, timeout=120)
        response.raise_for_status() # Raise exception for 4XX/5XX errors
        
        result = response.json()
        if "response" in result:
            raw_text = result["response"]
            # Parse ONLY the model's generated answer.
            # Unsloth renders Llama-3 headers as plain text with NO newline before 'assistant':
            # e.g. '...User Query: what?assistant\nHere is the answer.'
            # So we simply split on 'assistant\n' and take the LAST part.
            import re
            if "assistant\n" in raw_text:
                clean_text = raw_text.split("assistant\n")[-1].strip()
                clean_text = clean_text.replace("<|eot_id|>", "").strip()
                return clean_text
            # Fallback: Llama-3 raw special tokens (if skip_special_tokens=False in Colab)
            if "<|start_header_id|>assistant<|end_header_id|>" in raw_text:
                clean_text = raw_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
                return clean_text.replace("<|eot_id|>", "").strip()
            # Fallback: Mistral format
            if "[/INST]" in raw_text:
                clean_text = raw_text.split("[/INST]")[-1].strip()
                return clean_text
            return raw_text.strip()
        else:
            return f"Error parsing Colab response: {result}"
            
    except requests.exceptions.HTTPError as errh:
        logger.error(f"HTTP Error: {errh}")
        return f"API Error ({response.status_code}): Ensure your Colab notebook is actively running and the ngrok tunnel hasn't expired."
    except requests.exceptions.ConnectionError:
        return "⚠️ Connection Error: Could not connect to the ngrok URL. Make sure the Colab cell is running and you copied the exact link (e.g., https://abc.ngrok-free.app/generate)."
    except Exception as e:
        logger.error(f"Colab API Error: {str(e)}")
        return f"API Error: {str(e)}"

# 2. The Execution Loop (Adapted for PEFT Pipeline)
def run_adapted_agent(user_query: str) -> str:
    """
    Runs the LexGuard pipeline using the Fine-Tuned Local Server.
    Step 1: RAG Retrieval
    Step 2: Model Extraction 
    Step 3: Python Rule Risk Calculation
    """
    print(f"\n🧠 [ADAPTED MODEL] LexGuard starting audit for query: '{user_query}'")
    
    # STEP 0: Filter out greetings and non-contract queries
    greeting_words = {'hi', 'hello', 'hey', 'sup', 'yo', 'greetings', 'howdy', 'good morning', 'good afternoon', 'good evening'}
    query_lower = user_query.strip().lower().rstrip('!?.,')
    if query_lower in greeting_words or len(query_lower.split()) <= 2 and not any(w in query_lower for w in ['contract', 'clause', 'liability', 'risk', 'term', 'party', 'parties', 'agreement', 'lease', 'indemnif']):
        return "👋 Hello! I'm LexGuard, your AI Contract Compliance Auditor. Ask me a question about your contracts, such as:\n\n• *What are the termination conditions?*\n• *Is there any uncapped liability?*\n• *Who are the parties in the agreements?*"
    
    # STEP 1: RAG Retrieval
    # We will pass the full query to the semantic search store because it uses embeddings 
    # to find the conceptual match, which is more robust than naive keyword slicing.
    retrieved_context = retrieve_local_clauses(user_query)
    
    if not retrieved_context.strip() or "No evidence found" in retrieved_context:
        return f"Could not find any clauses related to your query to analyze. Please try a different query."
        
    # STEP 2: Model Extraction
    formatted_prompt = LLAMA3_PROMPT_TEMPLATE.format(
        context=retrieved_context,
        query=user_query
    )
    
    print(f"🤖 Sending retrieved context to Colab Ngrok Server ({COLAB_API_URL})...")
    model_extraction = query_colab_api(formatted_prompt)
    
    if "Error" in model_extraction or "⏳" in model_extraction:
        return model_extraction
        
    # STEP 3: Python Rule Risk Calculation
    print(f"⚖️ Running extracted facts through `calculate_risk_level` rules...")
    risk_assessment = calculate_risk_level(model_extraction)
    
    print("\n✅ Final Verdict Reached.")
    
    # Combine the model's extraction and the python risk assessment
    final_output = f"**Model Extraction:**\n{model_extraction}\n\n**Rule-Based Risk Assessment:**\n{risk_assessment}"
    return final_output


if __name__ == "__main__":
    test_query = "Audit the lease agreements to see if the pet deposit amount is compliant."
    final_output = run_adapted_agent(test_query)
    print(f"\n[FINAL OUTPUT]\n{final_output}")
