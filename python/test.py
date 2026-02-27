import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
from pipeline import LexGuardPipeline
from database import SnowflakeManager

def test_pipeline():
    file_path = "/Users/dudoan/Documents/UMKC/Big Data Analytics/project 2/CUAD_v1/full_contract_txt/LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT.txt"
    print(f"Testing pipeline with file: {file_path}")
    
    # 1. Init pipeline
    pipe = LexGuardPipeline()
    
    # 2. Test document loading
    doc_data = pipe.load_document(file_path)
    print("--- Loaded Document ---")
    print(f"Total Text length: {len(doc_data['full_text'])}")
    
    # 3. Test Chunking
    chunks_fixed = pipe.chunk_fixed(doc_data['full_text'])
    chunks_semantic = pipe.chunk_semantic(doc_data['full_text'])
    chunks_page = pipe.chunk_page(doc_data['pages'])
    
    print("--- Chunking Results ---")
    print(f"Fixed: {len(chunks_fixed)}")
    print(f"Semantic: {len(chunks_semantic)}")
    print(f"Page: {len(chunks_page)}")
    
    # 4. Test Indexing & Search
    pipe.build_index(chunks_semantic)
    
    query = "What is the governing law of this agreement?"
    results = pipe.search(query, top_k=2)
    print("--- Search Results ---")
    print(f"Found {len(results)} chunks for query: '{query}'")
    
    # 5. Test generation
    if results:
        texts = [r["text"] for r in results]
        answer = pipe.generate_answer(query, texts)
        print("--- Answer Generation ---")
        print(answer)
        
    # 6. Test Metadata Extraction (if Groq key works)
    if "GROQ_API_KEY" in os.environ:
        print("--- Testing Groq Metadata Extraction ---")
        meta = pipe.extract_metadata(doc_data['full_text'][:20000])
        print(f"Received JSON with {len(meta.keys())} keys")
        
    print("--- Completed Successfully ---")

if __name__ == "__main__":
    test_pipeline()

