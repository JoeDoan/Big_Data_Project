import os
import streamlit as st
import snowflake.connector

import config  # noqa: F401 — seeds randomness on import
from local_store import LocalStore

# Removed externalbrowser to use standard password auth that worked for ingest.py

@st.cache_resource(ttl=3600)
def get_snowflake_connection():
    print("❄️ Opening persistent Snowflake connection...")

    return snowflake.connector.connect(
        user=os.getenv("SNOW_USER"),
        password=os.getenv("SNOW_PASS"),
        account=os.getenv("SNOW_ACCOUNT"),
        role=os.getenv("SNOW_ROLE", "TRAINING_ROLE"),
        warehouse=os.getenv("SNOW_WH", "COMPUTE_WH"),
        database=os.getenv("SNOW_DB", "LEXGUARD_DB"),
        schema=os.getenv("SNOW_SCHEMA", "CONTRACT_DATA")
    )

def retrieve_contract_clauses(search_term: str) -> str:
    """
    Searches the Snowflake database for specific legal contract clauses based on a keyword.
    Use this tool whenever the user asks about the contents of the contracts.

    Args:
        search_term: A specific keyword or short phrase to search for (e.g., "termination", "liability").

    Returns:
        A string containing the retrieved contract chunks, or an error message if the search fails.
    """
    print(f"🔧 Tool Invoked: Searching Snowflake for '{search_term}'...")
    
    try:
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        
        query = f"""
            SELECT DOC_NAME, CHUNK_TEXT 
            FROM CONTRACT_CHUNKS 
            WHERE CHUNK_TEXT ILIKE '%{search_term}%'
            LIMIT 5;
        """
        cursor.execute(query)
        results = cursor.fetchall()
        
        if not results:
            return f"No evidence found in the contracts for '{search_term}'."
            
        evidence = []
        for row in results:
            evidence.append(f"[Source: {row[0]}]\n{row[1]}")
            
        return "\n\n---\n\n".join(evidence)
        
    except Exception as e:
        # Added this so we can see the exact raw error in your Mac terminal
        print(f"\n❌ RAW SNOWFLAKE ERROR: {str(e)}\n") 
        return f"Database error: {str(e)}"

def calculate_risk_level(clause_text: str) -> str:
    """
    Analyzes a specific contract clause to determine if it contains high-risk language.
    Use this tool if the user asks to evaluate risk or danger in a clause.

    Args:
        clause_text: The exact text of the legal clause to evaluate.

    Returns:
        A string indicating 'High Risk', 'Medium Risk', or 'Low Risk' with a brief reason.
    """
    print("🔧 Tool Invoked: Calculating risk level...")
    
    text_lower = clause_text.lower()
    if "indemnify" in text_lower or "immediate termination" in text_lower:
        return "High Risk: Contains indemnification or immediate termination clauses."
    elif "penalty" in text_lower or "breach" in text_lower:
        return "Medium Risk: Mentions penalties or breach conditions."
    else:
        return "Low Risk: No standard high-risk keywords detected."


def retrieve_local_clauses(search_term: str, top_k: int = 5) -> str:
    """
    Searches the local separated JSON stores for contract clauses by keyword.
    This is the offline/local alternative to retrieve_contract_clauses(), inspired
    by HyperGraphRAG's separated entity_vdb + hyperedge_vdb retrieval pattern.

    Args:
        search_term: A keyword or phrase to search the clause index for.
        top_k: Maximum number of results to return.

    Returns:
        A string containing the matched contract chunks, or a not-found message.
    """
    print(f"🔧 Tool Invoked: Searching local store for '{search_term}'...")

    try:
        store = LocalStore(working_dir=config.HYPERPARAMS["working_dir"])
        results = store.search_clauses(search_term, top_k=top_k)

        # Fallback to full text search if index returns nothing (e.g., for full sentences)
        if not results:
            print(f"   ℹ️ Index miss. Falling back to full text search...")
            all_chunks = store.get_all_chunks()
            
            # Clean common stop words from search words
            stop_words = {'tell', 'me', 'about', 'what', 'is', 'are', 'the', 'of', 'in', 'and', 'to', 'for', 'any'}
            search_words = [w.lower() for w in search_term.replace('?', '').replace('.', '').replace(',', '').replace("'", '').split() if w.lower() not in stop_words]
            
            scored_chunks = []
            for chunk in all_chunks:
                text_lower = chunk["text"].lower()
                
                # Base score: count keyword matches
                score = sum(1 for w in search_words if w in text_lower)
                
                # Boost specific structural phrases
                if "change of control" in search_term.lower() and "change of control" in text_lower:
                    score += 5
                if "merger" in search_term.lower() and "merger" in text_lower:
                    score += 5
                if "parti" in search_term.lower() and ("parties" in text_lower or "party" in text_lower or "between" in text_lower):
                    score += 5
                    
                scored_chunks.append((score, chunk))
            
            # Always return the highest-scoring chunks, even if scores are low, 
            # so the model has *some* context rather than failing completely.
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            results = [c[1] for c in scored_chunks[:top_k] if c[0] > 0] # Filter out strict 0-score chunks
            
            # If still nothing, just grab the first few chunks of the first document as a hard fallback
            if not results and all_chunks:
                 print(f"   ⚠️ Hard fallback to document opening...")
                 results = all_chunks[:top_k]

        if not results:
            return "" # Return empty string instead of english message so agent handles it

        evidence = []
        for r in results:
            evidence.append(f"[Source: {r['doc_name']}]\n{r['text']}")

        return "\n\n---\n\n".join(evidence)

    except Exception as e:
        print(f"\n❌ LOCAL STORE ERROR: {str(e)}\n")
        return ""