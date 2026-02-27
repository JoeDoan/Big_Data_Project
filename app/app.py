import torch
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
import uuid
import tempfile
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))

from pipeline import LexGuardPipeline
from database import SnowflakeManager
from rules_engine import RulesEngine

# Page config
st.set_page_config(page_title="LexGuard - Legal RAG System", layout="wide", page_icon="⚖️")

# Initialize Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "pipeline" not in st.session_state:
    st.session_state.pipeline = LexGuardPipeline()
if "db" not in st.session_state:
    st.session_state.db = SnowflakeManager()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "metadata" not in st.session_state:
    st.session_state.metadata = None
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "rules_engine" not in st.session_state:
    st.session_state.rules_engine = RulesEngine()
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# Sidebar setup
with st.sidebar:
    st.title("LexGuard Settings ⚙️")
    
    chunking_strategy = st.radio(
        "Select Chunking Strategy:",
        ["Fixed", "Page", "Semantic"],
        index=2,
        help="Fixed: exact character overlap. Page: splits by PDF pages. Semantic: splits by paragraphs."
    )
    
    st.divider()
    
    st.subheader("Document Upload 📄")
    uploaded_file = st.file_uploader("Upload a Contract (.txt, .pdf)", type=["txt", "pdf"])
    
if uploaded_file and uploaded_file.name != st.session_state.current_file:
        with st.spinner("Processing Document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            try:
                doc_data, num_chunks = st.session_state.pipeline.process_and_index(tmp_path, str(chunking_strategy))
                
                # Phase 2: Metadata Extraction via Gemini Flash
                if getattr(st.session_state.pipeline, "llm_client", None):
                    st.session_state.metadata = st.session_state.pipeline.extract_metadata()
                    # Phase 3: Neuro-Symbolic Logic Engine
                    st.session_state.analysis = st.session_state.rules_engine.analyze(st.session_state.metadata)
                else:
                    st.session_state.metadata = {"warning": "Gemini client missing. Skipping extraction."}
                    st.session_state.analysis = None
                
                st.session_state.doc_loaded = True
                st.session_state.current_file = uploaded_file.name
                st.session_state.chunk_count = num_chunks
                st.session_state.messages = [] 
                st.sidebar.success(f"Loaded '{uploaded_file.name}' into {num_chunks} {str(chunking_strategy).lower()} chunks!")
            except Exception as e:
                st.sidebar.error(f"Error processing document: {str(e)}")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)


# Main Content Area
st.title("⚖️ LexGuard RAG Co-Pilot")

if not st.session_state.doc_loaded:
    st.info("👈 Please upload a legal document in the sidebar to begin.")
else:
    # Display Metadata in an expander
    with st.expander("📊 Extracted Metadata & Risk Analysis (Gemini + Rules Engine)", expanded=True):
        if st.session_state.analysis and st.session_state.analysis.get("status") == "SUCCESS":
            flags = st.session_state.analysis.get("flags", [])
            if flags:
                st.subheader("🚩 Automated Risk Assessment")
                for flag in flags:
                    if flag["level"] == "CRITICAL":
                        st.error(f"**CRITICAL ({flag['field']}):** {flag['message']}")
                    elif flag["level"] == "WARNING":
                        st.warning(f"**WARNING ({flag['field']}):** {flag['message']}")
                    else:
                        st.info(f"**INFO ({flag['field']}):** {flag['message']}")
            else:
                st.success("✅ No critical legal risks or red flags detected by the logic engine.")
                
            st.divider()
            st.subheader("📝 Structured Metadata")
            st.json(st.session_state.metadata)
        elif st.session_state.metadata and "error" in st.session_state.metadata:
            st.warning(st.session_state.metadata["error"])
        elif st.session_state.metadata and "warning" in st.session_state.metadata:
            st.warning(st.session_state.metadata["warning"])
        
        st.write(f"**Indexed Chunks:** {st.session_state.chunk_count} ({chunking_strategy} strategy)")

    st.divider()
    
    # Chat Area
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            if "metrics" in msg:
                st.divider()
                st.markdown("##### 📊 Query Performance Dashboard")
                m1, m2, m3 = st.columns(3)
                m1.metric(label="⏱️ Response Time", value=f"{msg['metrics']['response_time']:.2f}s")
                m2.metric(label="🎯 Precision (Simulated)", value=f"{msg['metrics']['precision']}%")
                m3.metric(label="🔄 Recall (Simulated)", value=f"{msg['metrics']['recall']}%")
                st.caption("*Note: Precision and Recall are simulated for UI demonstration.*")
                
            if "evidence" in msg:
                with st.expander("Supporting Evidence"):
                    for idx, chunk in enumerate(msg["evidence"]):
                        st.markdown(f"**Chunk {idx+1} [Score: {chunk['score']:.2f}]**")
                        st.write(chunk["text"])
                        st.divider()

    # User Input
    if prompt := st.chat_input(f"Ask questions about {st.session_state.current_file}..."):
        # Display user msg
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Answer generation
        with st.chat_message("assistant"):
            import time
            start_time = time.time()
            with st.spinner("Searching and generating answer..."):
                retrieved_chunks = st.session_state.pipeline.search(prompt, top_k=3)
                
                if not retrieved_chunks:
                    response = "I couldn't find any relevant clauses in the document to answer that."
                else:
                    evidence_texts = [c["text"] for c in retrieved_chunks]
                    response = st.session_state.pipeline.generate_answer(prompt, evidence_texts)
                
                end_time = time.time()
                response_time = end_time - start_time
                
                # Show Response 
                st.markdown(response)
                
                # Show Metrics Dashboard
                st.divider()
                st.markdown("##### 📊 Query Performance Dashboard")
                m1, m2, m3 = st.columns(3)
                m1.metric(label="⏱️ Response Time", value=f"{response_time:.2f}s")
                # For demo purposes, we display mock/placeholder values. 
                # In a real system, calculating Precision/Recall dynamically requires an LLM-as-a-judge or ground truth.
                m2.metric(label="🎯 Precision (Simulated)", value="89.5%")
                m3.metric(label="🔄 Recall (Simulated)", value="84.2%")
                st.caption("*Note: Precision and Recall are simulated for UI demonstration.*")
                
                # Show Evidence
                if retrieved_chunks:
                    with st.expander("Supporting Evidence"):
                        for idx, chunk in enumerate(retrieved_chunks):
                            st.markdown(f"**Chunk {idx+1} [Score: {chunk['score']:.2f}]**")
                            st.write(chunk["text"])
                            st.divider()
                            
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "evidence": retrieved_chunks,
                    "metrics": {
                        "response_time": response_time,
                        "precision": 89.5,
                        "recall": 84.2
                    }
                })
                
                # Snowflake tracking background task
                if st.session_state.db.is_connected:
                    st.session_state.db.save_chat(
                        st.session_state.session_id,
                        st.session_state.current_file or "",
                        prompt,
                        response,
                        retrieved_chunks
                    )
