import os
import time
import logging
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Import the tools we defined in tools.py
from tools import retrieve_contract_clauses, calculate_risk_level

# Configure logging for the Streamlit terminal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API Key from .env
load_dotenv()
# Initialize the NEW Google GenAI client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 1. The System Prompt
SYSTEM_PROMPT = """
You are LexGuard, a Neuro-Symbolic Compliance Auditor. Your job is to audit Residential Lease Agreements against strict compliance rules.
You operate on a 'Recall-Then-Reason' pipeline:
1. First, use 'retrieve_contract_clauses' to search the Snowflake database for specific legal clauses by keyword.
2. Second, use 'calculate_risk_level' to evaluate whether a clause contains high-risk language.

Never guess or assume contract details. Always use your tools. If a tool returns an error, read the error and try a different search term or query. Once you have gathered all necessary context, provide a final compliance verdict explaining if the clause passes or fails.
"""

# Map string names to the actual Python functions so the loop can call them
AVAILABLE_TOOLS = {
    "retrieve_contract_clauses": retrieve_contract_clauses,
    "calculate_risk_level": calculate_risk_level
}

# 2. The Execution Loop
def run_lexguard_agent(user_query: str) -> dict:
    """
    Runs the multi-step reasoning loop for LexGuard.
    Handles tool routing, captures errors, and prevents infinite loops.

    Returns:
        dict with keys: response (str), trace (list[dict]), tool_calls (list[str]),
                        retrieval_count (int), risk_level (str), success (bool)
    """

    trace = []       # Structured execution trace
    tool_names = []  # Names of tools called
    retrieval_count = 0
    risk_level = "N/A"

    # Start a chat session using the new SDK configuration
    chat = client.chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=list(AVAILABLE_TOOLS.values()),
            temperature=0.1, # Keep temperature low for analytical tasks
        )
    )

    max_steps = 5  # Prevent infinite loops
    current_step = 0

    print(f"\n🧠 LexGuard starting audit for query: '{user_query}'")
    trace.append({"step": "start", "detail": f"Query: {user_query}"})

    # The initial prompt is the user's question
    prompt = user_query

    while current_step < max_steps:
        current_step += 1
        step_start = time.time()

        try:
            # Send the prompt (or the tool results) to Gemini
            response = chat.send_message(prompt)
        except Exception as e:
            error_msg = f"API Error communicating with Gemini: {str(e)}"
            logger.error(error_msg)
            trace.append({"step": "error", "detail": error_msg, "time": 0})
            return {"response": error_msg, "trace": trace, "tool_calls": tool_names,
                    "retrieval_count": retrieval_count, "risk_level": risk_level, "success": False}

        step_elapsed = round(time.time() - step_start, 2)

        # Check if Gemini decided it needs to use a tool (New SDK syntax)
        if response.function_calls:
            tool_responses = [] # We need to collect the results to send back

            for tool_call in response.function_calls:
                tool_name = tool_call.name
                tool_args = tool_call.args

                print(f"🛠️ Agent called tool: '{tool_name}' with arguments: {tool_args}")

                # Execute the tool and catch any errors
                tool_start = time.time()
                if tool_name in AVAILABLE_TOOLS:
                    try:
                        tool_result = AVAILABLE_TOOLS[tool_name](**tool_args)
                    except Exception as e:
                        tool_result = f"Error executing {tool_name}: {str(e)}"
                else:
                    tool_result = f"Error: Tool '{tool_name}' not found."

                tool_elapsed = round(time.time() - tool_start, 2)
                tool_names.append(tool_name)

                # Track retrieval count
                if tool_name == "retrieve_contract_clauses":
                    retrieval_count += str(tool_result).count("[Source:")
                # Track risk level
                if tool_name == "calculate_risk_level":
                    if "High Risk" in str(tool_result):
                        risk_level = "High"
                    elif "Medium Risk" in str(tool_result):
                        risk_level = "Medium"
                    else:
                        risk_level = "Low"

                trace.append({
                    "step": f"tool_call",
                    "tool": tool_name,
                    "args": str(tool_args),
                    "result_preview": str(tool_result)[:150],
                    "time": tool_elapsed
                })

                # Format the response exactly how the new SDK requires it
                tool_responses.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": str(tool_result)}
                    )
                )

            # Set the prompt for the next loop iteration to be the tool results
            prompt = tool_responses

        else:
            # If there are no function calls, Gemini gave us the final text!
            print("\n✅ Final Verdict Reached.")
            trace.append({"step": "response", "detail": "Final verdict generated", "time": step_elapsed})
            return {"response": response.text, "trace": trace, "tool_calls": tool_names,
                    "retrieval_count": retrieval_count, "risk_level": risk_level, "success": True}

    # If it hits max steps without returning, force it to stop
    timeout_msg = "⚠️ LexGuard Audit failed: Maximum reasoning steps exceeded."
    logger.error(timeout_msg)
    trace.append({"step": "timeout", "detail": timeout_msg, "time": 0})
    return {"response": timeout_msg, "trace": trace, "tool_calls": tool_names,
            "retrieval_count": retrieval_count, "risk_level": risk_level, "success": False}

# Test the loop directly in the terminal
if __name__ == "__main__":
    test_query = "Audit the lease agreements to see if the pet deposit amount is compliant."
    result = run_lexguard_agent(test_query)
    print(f"\n[FINAL OUTPUT]\n{result['response']}")
    print(f"\n[TRACE]\n{result['trace']}")