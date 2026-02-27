import os
import json
import re
import ssl
import torch
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pypdf
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# macOS SSL Certificate workaround for downloading models
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


class LexGuardPipeline:
    def __init__(self, use_local_llm: bool = True):
        """Initialize models and RAG pipeline."""
        print("Initializing Embedding Model (RAG Backbone)...")
        try:
            print("Initializing Embedding Model (RAG Backbone)...")
            self.embedder = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
            self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        except Exception as e:
            print(f"Warning: Failed to load FAISS embedder: {e}")
            self.embedder = None
        print("Initializing Cross-Encoder Reranker...")
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception as e:
            print(f"Warning: Failed to load cross-encoder. Falling back to dense-only. {e}")
            self.reranker = None
        
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if self.gemini_api_key:
            from google import genai
            self.llm_client = genai.Client(api_key=self.gemini_api_key)
            print("✅ Gemini Client initialized successfully.")
        else:
            self.llm_client = None
            print("❌ Warning: GEMINI_API_KEY not found in .env. LLM Extraction will fail.")
            
        self.groq_api_key = os.environ.get("GROQ_API_KEY")
        if self.groq_api_key:
            from groq import Groq
            self.groq_client = Groq(api_key=self.groq_api_key)
            print("✅ Groq Client initialized successfully.")
        else:
            self.groq_client = None
            print("❌ Warning: GROQ_API_KEY not found in .env.")
                
        # State
        self.chunks: List[str] = []
        self.faiss_index: Any = None
        self.bm25: Any = None

    def load_document(self, file_path: str) -> Dict[str, Any]:
        """Loads a document (txt or pdf) and returns mapping of page numbers to text."""
        pages = {}
        if file_path.lower().endswith(".pdf"):
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    pages[i] = page.extract_text() or ""
        elif file_path.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
                simulated_page_size = 3000
                for i in range(0, len(text), simulated_page_size):
                    pages[i // simulated_page_size] = text[i:i+simulated_page_size]
        else:
            raise ValueError("Unsupported file format. Please use .txt or .pdf")
            
        full_text = "\n\n".join(pages.values())
        return {
            "full_text": full_text,
            "pages": pages,
            "filename": os.path.basename(file_path)
        }

    def chunk_fixed(self, text: str, size: int = 900, overlap: int = 150) -> List[str]:
        """Fixed character chunking."""
        chunks = []
        step = max(1, size - overlap)
        for i in range(0, len(text), step):
            c = text[i:i+size].strip()
            if len(c) > 50:
                chunks.append(c)
        return chunks

    def chunk_page(self, pages: Dict[int, str]) -> List[str]:
        """One chunk per page."""
        chunks = []
        for pnum, ptext in pages.items():
            cleaned = ptext.strip()
            if len(cleaned) > 50:
                chunks.append(cleaned)
        return chunks

    def chunk_semantic(self, text: str) -> List[str]:
        """Paragraph-based semantic chunking."""
        paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
        merged, buf = [], ""
        for p in paras:
            if len(buf) < 500:
                buf = (buf + "\n\n" + p).strip()
            else:
                merged.append(buf)
                buf = p
        if buf:
            merged.append(buf)
        return [m for m in merged if len(m) > 100]

    def build_index(self, chunks: List[str]):
        """Builds FAISS index for retrieved chunks."""
        self.chunks = chunks
        if not chunks:
            self.faiss_index = None
            return
            
        print(f"Embedding {len(chunks)} chunks...")
        embeddings = self.embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings)
        print("FAISS Index build complete.")
        
        print("Building BM25 Sparse Index...")
        try:
            from rank_bm25 import BM25Okapi
            tokenized_chunks = [chunk.lower().split() for chunk in chunks]
            self.bm25 = BM25Okapi(tokenized_chunks)
            print("BM25 Sparse Index build complete.")
        except ImportError:
            print("Warning: rank_bm25 not installed. Sparse retrieval disabled.")
            self.bm25 = None

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Hybrid Search (FAISS + BM25) with Cross-Encoder Reranking."""
        if not self.faiss_index or not self.chunks:
            return []
            
        candidates = set()
        
        # 1. Dense Retrieval (FAISS)
        q_emb = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        dense_scores, dense_idx = self.faiss_index.search(q_emb, top_k * 2)
        
        for i in dense_idx[0]:
            if i < len(self.chunks) and i >= 0:
                candidates.add(int(i))
                
        # 2. Sparse Retrieval (BM25)
        if self.bm25:
            tokenized_query = query.lower().split()
            sparse_scores = self.bm25.get_scores(tokenized_query)
            sparse_idx = np.argsort(sparse_scores)[::-1][:top_k * 2]
            for i in sparse_idx:
                 if i < len(self.chunks) and sparse_scores[i] > 0 and i >= 0:
                     candidates.add(int(i))
                     
        if not candidates:
            return []
            
        candidate_list = list(candidates)
        chunks_to_rerank = [self.chunks[i] for i in candidate_list]
        
        # 3. Reranking (Cross-Encoder)
        if self.reranker:
            pairs = [[query, chunk] for chunk in chunks_to_rerank]
            rerank_scores = self.reranker.predict(pairs)
            
            # Sort by rerank score
            ranked_indices = np.argsort(rerank_scores)[::-1][:top_k]
            
            results = []
            for idx in ranked_indices:
                orig_idx = candidate_list[idx]
                results.append({
                    "chunk_id": orig_idx, 
                    "score": float(rerank_scores[idx]), 
                    "text": chunks_to_rerank[idx]
                })
            return results
            
        # Fallback if no reranker
        return [{"chunk_id": i, "score": float(dense_scores[0][list(dense_idx[0]).index(i)] if i in dense_idx[0] else 0.5), "text": self.chunks[i]} for i in list(candidates)[:top_k]]

    def generate_answer(self, query: str, evidence_chunks: List[str]) -> str:
        """Generates conversational answer using Gemini and RAG evidence."""
        if not self.llm_client:
            return "Gemini client not initialized. Cannot generate answer."
            
        evidence = "\n\n".join([f"[Chunk {i+1}] {chunk}" for i, chunk in enumerate(evidence_chunks)])
        prompt = f"""Answer the question using ONLY the evidence below. Make your answer concise.

Evidence:
{evidence}

Question:
{query}

Answer:"""
        try:
            from google.genai import types
            response = self.llm_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3
                )
            )
            return response.text or "No response generated."
        except Exception as e:
            return f"Error generating answer: {e}"

    def extract_metadata(self, doc_text: str = None) -> Dict[str, Any]:
        """Phase 1 & 2: Multi-Agent RAG Extraction via Gemini (gemini-3-flash-preview)."""
        if not getattr(self, "llm_client", None):
            return {"error": "Gemini client not initialized. Check GEMINI_API_KEY."}

        import concurrent.futures
        import json

        batches = [
            {
                "name": "Batch 1: Core Terms",
                "query": "Document Name, Parties, Agreement Date, Effective Date, Expiration Date, Renewal Term, Notice Period To Terminate Renewal, Governing Law, Most Favored Nation, Competitive Restriction Exception",
                "fields": [
                    "Document Name", "Parties", "Agreement Date", "Effective Date", 
                    "Expiration Date", "Renewal Term", "Notice Period To Terminate Renewal", 
                    "Governing Law", "Most Favored Nation", "Competitive Restriction Exception"
                ]
            },
            {
                "name": "Batch 2: Covenants & Terms",
                "query": "Non-Compete, Exclusivity, No-Solicit Of Customers, No-Solicit Of Employees, Non-Disparagement, Termination For Convenience, Rofr/Rofo/Rofn, Change Of Control, Anti-Assignment, Revenue/Profit Sharing",
                "fields": [
                    "Non-Compete", "Exclusivity", "No-Solicit Of Customers", "No-Solicit Of Employees", 
                    "Non-Disparagement", "Termination For Convenience", "Rofr/Rofo/Rofn", 
                    "Change Of Control", "Anti-Assignment", "Revenue/Profit Sharing"
                ]
            },
            {
                "name": "Batch 3: IP & Licensing",
                "query": "Price Restrictions, Minimum Commitment, Volume Restriction, Ip Ownership Assignment, Joint Ip Ownership, License Grant, Non-Transferable License, Affiliate License-Licensor, Affiliate License-Licensee, Unlimited/All-You-Can-Eat-License, Irrevocable Or Perpetual License",
                "fields": [
                    "Price Restrictions", "Minimum Commitment", "Volume Restriction", 
                    "Ip Ownership Assignment", "Joint Ip Ownership", "License Grant", 
                    "Non-Transferable License", "Affiliate License-Licensor", "Affiliate License-Licensee", 
                    "Unlimited/All-You-Can-Eat-License", "Irrevocable Or Perpetual License"
                ]
            },
            {
                "name": "Batch 4: Liability & Audit",
                "query": "Source Code Escrow, Post-Termination Services, Audit Rights, Uncapped Liability, Cap On Liability, Liquidated Damages, Warranty Duration, Insurance, Covenant Not To Sue, Third Party Beneficiary",
                "fields": [
                    "Source Code Escrow", "Post-Termination Services", "Audit Rights", 
                    "Uncapped Liability", "Cap On Liability", "Liquidated Damages", 
                    "Warranty Duration", "Insurance", "Covenant Not To Sue", "Third Party Beneficiary"
                ]
            }
        ]

        def process_batch(batch):
            print(f"[{batch['name']}] Performing Targeted RAG Retrieval (top_k=25)...")
            retrieved_chunk_indices = set()
            if self.faiss_index and self.chunks:
                results = self.search(batch["query"], top_k=25)
                for r in results:
                    chunk_idx = r.get('chunk_id')
                    if chunk_idx is not None:
                        retrieved_chunk_indices.add(chunk_idx)
                        if chunk_idx > 0:
                            retrieved_chunk_indices.add(chunk_idx - 1)
                        if chunk_idx < len(self.chunks) - 1:
                            retrieved_chunk_indices.add(chunk_idx + 1)
                
                sorted_indices = sorted(list(retrieved_chunk_indices))
                context_pieces = [f"[Chunk {idx}] {self.chunks[idx]}" for idx in sorted_indices]
                rag_context = "\n\n[...] ".join(context_pieces)
                print(f"[{batch['name']}] -> Extracted {len(sorted_indices)} padded chunks.")
            else:
                rag_context = doc_text[:20000] if doc_text else ""
                print(f"[{batch['name']}] -> Warning: FAISS index not built, using raw text.")

            fields_list_str = "\n".join([f"- {f}" + (" (Yes/No/None)" if f not in ["Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date", "Renewal Term", "Notice Period To Terminate Renewal", "Governing Law"] else "") for f in batch["fields"]])
            
            system_prompt = f"""You are a highly precise legal AI agent acting as a data extractor. 
Your ONLY job is to extract values for exactly {len(batch['fields'])} predefined metadata fields from the provided legal contract context.

Rules:
1. Return ONLY a valid JSON object. No explanations, no markdown formatting (like ```json).
2. The JSON keys MUST EXACTLY MATCH the field names listed below.
3. If a field is not mentioned or if there's no clear evidence in the text, you MUST return "None mentioned". For Yes/No/None fields, ALWAYS return EXACTLY "Yes", "No", or "None". DO NOT use True/False.
4. DO NOT make assumptions or infer values not present in the text.
5. For Parties, list all extracting names. For dates, standardize if possible.

Required Fields:
{fields_list_str}
"""

            user_prompt = f"""Extract the {len(batch['fields'])} fields from this contract text.

Contract Text Context:
{rag_context}
"""
            for attempt in range(3):
                try:
                    if attempt > 0:
                        print(f"[{batch['name']}] Retrying extraction via Gemini (Attempt {attempt+1}/3)...")
                    else:
                        print(f"[{batch['name']}] Generating extraction via Gemini (gemini-3-flash-preview)...")
                        
                    from google.genai import types
                    response = self.llm_client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[system_prompt, user_prompt],
                        config=types.GenerateContentConfig(
                            temperature=0.1,
                            response_mime_type="application/json"
                        )
                    )
                    content = response.text
                    if not content:
                        if attempt == 2:
                            return {f: "Error: Empty response" for f in batch["fields"]}
                        continue
                    return json.loads(content)
                except Exception as e:
                    import time
                    time.sleep(2)
                    if attempt == 2:
                        print(f"[{batch['name']}] Error: {e}")
                        return {f: f"Error: {str(e)}" for f in batch["fields"]}

        final_results = {}
        print("Starting Multi-Agent Sequential Extraction (4 batches)...")
        for batch in batches:
            batch_result = process_batch(batch)
            if isinstance(batch_result, dict):
                final_results.update(batch_result)
            else:
                print("Unexpected result type from batch processing.")
        
        return final_results

    def process_and_index(self, file_path: str, chunking_strategy: str = "Semantic"):
        """Phase 1: Loads and chunks a document, then builds FAISS index."""
        doc_data = self.load_document(file_path)
        
        if chunking_strategy == "Fixed":
            chunks = self.chunk_fixed(doc_data["full_text"])
        elif chunking_strategy == "Page":
            chunks = self.chunk_page(doc_data["pages"])
        else: # Semantic default
            chunks = self.chunk_semantic(doc_data["full_text"])
            
        self.build_index(chunks)
        return doc_data, len(chunks)
