import streamlit as st
import os

# Import our working agent loops
from agent import run_lexguard_agent as run_baseline_agent
from adapted_agent import run_adapted_agent

# 1. Page Configuration
st.set_page_config(page_title="LexGuard Auditor", page_icon="⚖️", layout="centered")
st.title("⚖️ LexGuard Compliance Auditor")
st.markdown("A Neuro-Symbolic Agent for auditing Residential Lease Agreements.")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.markdown("### AI Model Selection")
    st.markdown("Select which pipeline LexGuard should use for your audit:")
    pipeline_choice = st.radio(
        "Audit Pipeline:",
        ("Baseline (Gemini API)", "Adapted (Mistral PEFT)")
    )
    
    if pipeline_choice == "Adapted (Mistral PEFT)":
        st.info("🧠 **Domain-Adapted Mode:** Using your custom-trained LoRA adapter hosted on HuggingFace to extract legal facts, followed by deterministic Python risk rules.")
    else:
        st.info("🔮 **Baseline Mode:** Using the generic Gemini 2.5 Flash model with standard RAG retrieval.")
        
    st.markdown("---")
    
    st.markdown("### 📊 System Status")

# 3. Initialize Conversation History (Rubric Requirement)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am LexGuard. What contract clauses would you like me to audit today?"}
    ]

# 4. Display Chat History (Rubric Requirement)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. User Chat Input (Rubric Requirement)
if prompt := st.chat_input("e.g., Are there any high-risk indemnification clauses?"):
    
    # Immediately display the user's question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 6. Loading Indicator & Agent Execution (Rubric Requirement)
    # Send the query to the selected pipeline
    with st.chat_message("assistant"):
        with st.spinner(f"LexGuard is analyzing using {pipeline_choice}..."):
            if pipeline_choice == "Adapted (Mistral PEFT)":
                response = run_adapted_agent(prompt)
            else:
                response = run_baseline_agent(prompt)
            
        # Display the final verdict
        st.markdown(response)
        
    # Save the agent's response to history
    st.session_state.messages.append({"role": "assistant", "content": response})