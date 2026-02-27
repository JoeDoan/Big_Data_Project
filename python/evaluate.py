import os
import random
import pandas as pd
import json
from pipeline import LexGuardPipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
CSV_PATH = "/Users/dudoan/Documents/UMKC/Big Data Analytics/project 2/CUAD_v1/master_clauses.csv"
TXT_DIR = "/Users/dudoan/Documents/UMKC/Big Data Analytics/project 2/CUAD_v1/full_contract_txt/"
NUM_SAMPLES = 1

# Map Groq extracted fields to CSV columns
# Note: Groq fields are typically capitalized/formatted differently.
# We'll map the expected JSON keys from Groq to the corresponding column names in the CSV.
# (Most columns follow a "Field Name-Answer" or just "Field Name" format. We will check both.)

def evaluate_extraction():
    print(f"Loading CSV data from {CSV_PATH}...")
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Init pipeline
    print("Initializing LexGuard pipeline...")
    pipe = LexGuardPipeline()
    if not pipe.llm_client:
        print("Error: Gemini client not initialized. Cannot proceed with extraction or judging.")
        return

    from groq import Groq
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        print("Error: GROQ_API_KEY not found in environment.")
        return
    groq_client = Groq(api_key=groq_api_key)

    # Get all .txt files
    all_txt_files = [f for f in os.listdir(TXT_DIR) if f.endswith(".txt")]
    if not all_txt_files:
        print(f"No .txt files found in {TXT_DIR}")
        return

    # Shuffle all .txt files to pick randomly
    random.shuffle(all_txt_files)
    
    total_docs = 0
    total_fields_extracted = 0
    total_matches = 0
    total_mismatches = 0
    total_not_found_in_csv = 0
    
    print(f"\n--- Starting Evaluation for {NUM_SAMPLES} randomly selected files ---\n")

    for file_name in all_txt_files:
        if total_docs >= NUM_SAMPLES:
            break
            
        txt_path = os.path.join(TXT_DIR, file_name)
        
        # Determine the corresponding PDF filename in the CSV
        base_name = file_name.replace(".txt", "")
        pdf_name_csv = base_name + ".pdf"
        # Check if the exact txt base name exists as pdf, or maybe with .PDF
        
        # Try finding the row in CSV
        # Filename column might have .pdf or .PDF or .txt
        row = df[(df['Filename'].str.lower() == pdf_name_csv.lower()) | (df['Filename'].str.lower() == file_name.lower())]
        
        if row.empty:
            print(f"[-] Document '{file_name}' not found in CSV. Skipping.")
            continue
            
        print(f"\n[+] Evaluating Document: {file_name}")
        
        # Read text and build FAISS Index for RAG
        try:
            print("    Chunking and indexing document (RAG)...")
            doc_data, num_chunks = pipe.process_and_index(txt_path, chunking_strategy="Fixed")
            doc_text = doc_data.get("full_text", "")
            print(f"    -> Created {num_chunks} fixed-size chunks.")
        except Exception as e:
            print(f"Error reading and indexing file {file_name}: {e}")
            continue
            
        # Extract metadata via Gemini API
        print("    Extracting metadata via Gemini API...")
        try:
            extracted_data = pipe.extract_metadata(doc_text)
        except Exception as e:
            extracted_data = {"error": f"Exception during Gemini Extraction: {str(e)}"}

        if not extracted_data or "error" in extracted_data:
            err_msg = extracted_data['error'] if extracted_data else 'Unknown'
            print(f"    Error during extraction: {err_msg}")
            continue
            
        total_docs += 1

        doc_matches = 0
        doc_mismatches = 0
        doc_not_found = 0
        
        matched_fields_list = []
        mismatched_fields_list = []

        print(f"    Preparing fields for LLM evaluation...")
        
        fields_to_evaluate = []
        for field, extracted_val in extracted_data.items():
            col_name_answer = f"{field}-Answer"
            col_name_regular = field
            
            csv_val = None
            if col_name_answer in df.columns:
                csv_val = row.iloc[0][col_name_answer]
            elif col_name_regular in df.columns:
                csv_val = row.iloc[0][col_name_regular]
            else:
                doc_not_found += 1
                continue
                
            if pd.isna(csv_val):
                csv_val_str = "None mentioned"
            else:
                csv_val_str = str(csv_val).strip()
            
            extracted_val_str = str(extracted_val).strip()
            fields_to_evaluate.append({
                "field": field,
                "extracted": extracted_val_str,
                "ground_truth": csv_val_str
            })
            total_fields_extracted += 1

        print("    Calling Groq API (openai/gpt-oss-20b judge)...")
        
        system_prompt = """You are an expert legal contract evaluator.
You are given a list of fields extracted from a contract by an AI, along with the "Ground Truth" values from human annotators.
Your job is to determine if the extracted value materially captures the ground truth meaning.
Do NOT be overly strict about exact string matching. Focus on SEMANTIC equivalence.

RULES FOR MATCHING:
1. "None mentioned", "Not found", "[]", "No", "None", and "N/A" all generally mean the clause is absent. If both ground truth and extracted indicate absence, it is a MATCH.
2. If Ground truth is "Yes" and extracted is a detailed explanation that confirms the clause exists, it is a MATCH.
3. If the extracted text contains the ground truth text but adds extra context (e.g. GT: "Virginia", Extracted: "Arbitration in McLean, Virginia"), it is a MATCH.
4. If Ground truth is a list of parties, and the extracted list contains the core entities even if missing minor affiliates or using abbreviations, it is a MATCH.
5. If the extracted date is fully spelled out but the ground truth is MM/DD/YY, it is a MATCH.

Return ONLY a valid JSON object. The keys must be the exact field names, and the value for each key must be an object with two properties:
- "match": true or false
- "reasoning": a brief 1-sentence explanation of why it matches or not."""

        user_prompt_lines = []
        for item in fields_to_evaluate:
            user_prompt_lines.append(f"Field: {item['field']}\nExtracted: {item['extracted']}\nGround Truth: {item['ground_truth']}\n---")
            
        user_prompt = "\n".join(user_prompt_lines)
        
        llm_results = {}
        
        for attempt in range(5):
            try:
                eval_completion = groq_client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                eval_content = eval_completion.choices[0].message.content
                if eval_content:
                    llm_results = json.loads(eval_content)
                break
            except Exception as e:
                wait_time = (attempt + 1) * 2
                print(f"    Judge API error/rate limit ({e}). Waiting {wait_time}s...")
                import time
                time.sleep(wait_time)
                
        if not llm_results:
            print("    Failed to get evaluation from LLM judge.")
            continue
            
        for item in fields_to_evaluate:
            field = item["field"]
            extracted = item["extracted"]
            csv_val = item["ground_truth"]
            
            judge_data = llm_results.get(field)
            if not isinstance(judge_data, dict):
                judge_data = {}
            is_match = judge_data.get("match", False)
            reasoning = judge_data.get("reasoning", "No reasoning provided.")
            
            if is_match:
                doc_matches += 1
                total_matches += 1
                matched_fields_list.append(field)
            else:
                doc_mismatches += 1
                total_mismatches += 1
                mismatched_fields_list.append(
                    f"{field}\n            CSV: '{csv_val[:80]}...'\n            Extracted: '{extracted[:80]}...'\n            Judge: {reasoning}"
                )

        print(f"    Results for {base_name}: {doc_matches} matches, {doc_mismatches} mismatches, {doc_not_found} fields not found in CSV.")
        
        if matched_fields_list:
            print("      [✓] Matched Fields:")
            # Batch them into a single line or multiple to not overwhelm standard output
            print("          " + ", ".join(matched_fields_list))
            
        if mismatched_fields_list:
            print("      [x] Mismatched Fields:")
            for m in mismatched_fields_list:
                print(f"          - {m}")

    print("\n--- Final Evaluation Summary ---")
    print(f"Total Documents Evaluated: {total_docs}")
    print(f"Total Fields Compared: {total_fields_extracted}")
    print(f"Total Matches: {total_matches}")
    print(f"Total Mismatches: {total_mismatches}")
    
    if total_fields_extracted > 0:
        accuracy = (total_matches / total_fields_extracted) * 100
        print(f"Overall Accuracy (Approximate): {accuracy:.2f}%")
    else:
        print("No valid fields could be compared.")

if __name__ == "__main__":
    evaluate_extraction()
