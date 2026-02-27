import os
import pandas as pd
import json
from tqdm import tqdm

def format_system_prompt():
    return """You are an expert legal assistant that extracts structured data from contracts. Output ONLY valid JSON.
Pay close attention to clauses that require a simple "Yes" if they exist, or "None mentioned" if they do not.
For example, if the contract says "Licensee shall not assign this agreement", then "Anti-Assignment" is "Yes".
If the contract says "Company grants an exclusive license", then "Exclusivity" is "Yes" and "License Grant" is "Yes"."""

def create_finetuning_dataset():
    # Paths
    csv_path = "../CUAD_v1/master_clauses.csv"
    txt_dir = "../CUAD_v1/full_contract_txt/"
    output_path = "finetune_data.jsonl"

    print(f"Loading CSV from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find CSV at {csv_path}")
        return
    print(f"Loaded {len(df)} rows from CSV.")

    # Build a set of all actual filenames (lowercased for lookup)
    all_txt_files = {f: f for f in os.listdir(txt_dir) if f.endswith('.txt')}
    all_txt_lower = {f.lower(): f for f in all_txt_files}
    print(f"Found {len(all_txt_files)} .txt files in folder.")

    # 41 target fields
    fields_list = [
        "Document Name", "Parties", "Agreement Date", "Effective Date",
        "Expiration Date", "Renewal Term", "Notice Period To Terminate Renewal",
        "Governing Law", "Most Favored Nation", "Competitive Restriction Exception",
        "Non-Compete", "Exclusivity", "No-Solicit Of Customers",
        "No-Solicit Of Employees", "Non-Disparagement", "Termination For Convenience",
        "Rofr/Rofo/Rofn", "Change Of Control", "Anti-Assignment",
        "Revenue/Profit Sharing", "Price Restrictions", "Minimum Commitment",
        "Volume Restriction", "Ip Ownership Assignment", "Joint Ip Ownership",
        "License Grant", "Non-Transferable License", "Affiliate License-Licensor",
        "Affiliate License-Licensee", "Unlimited/All-You-Can-Eat-License",
        "Irrevocable Or Perpetual License", "Source Code Escrow",
        "Post-Termination Services", "Audit Rights", "Uncapped Liability",
        "Cap On Liability", "Liquidated Damages", "Warranty Duration",
        "Insurance", "Covenant Not To Sue", "Third Party Beneficiary"
    ]

    valid_documents = 0
    not_found = []

    with open(output_path, "w", encoding="utf-8") as f:
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Documents"):
            pdf_filename = str(row['Filename']).strip()
            
            # Strategy 1: Replace .pdf -> .txt directly (exact 1-to-1 match)
            txt_name = pdf_filename.replace('.pdf', '.txt').replace('.PDF', '.txt')
            
            actual_fname = None
            # Strategy 2: Exact match
            if txt_name in all_txt_files:
                actual_fname = txt_name
            # Strategy 3: Case-insensitive match
            elif txt_name.lower() in all_txt_lower:
                actual_fname = all_txt_lower[txt_name.lower()]
            # Strategy 4: Try stripping/adding spaces
            elif txt_name.strip() in all_txt_files:
                actual_fname = txt_name.strip()
            else:
                # Strategy 5: Try prefix match (for trailing space filename variations like "Agreement .txt")
                name_without_ext = txt_name.replace('.txt', '').strip()
                for key_lower, key_actual in all_txt_lower.items():
                    if key_lower.replace('.txt', '').strip() == name_without_ext.lower():
                        actual_fname = key_actual
                        break

            if actual_fname is None:
                not_found.append(pdf_filename)
                continue

            txt_filepath = os.path.join(txt_dir, actual_fname)
            try:
                with open(txt_filepath, 'r', encoding='utf-8') as tf:
                    doc_text = tf.read()[:20000]
            except Exception as e:
                not_found.append(pdf_filename)
                continue

            if not doc_text.strip():
                not_found.append(pdf_filename)
                continue

            # Build ground truth JSON from CSV
            expected_output = {}
            for field in fields_list:
                col_name_answer = f"{field}-Answer"
                csv_val = None
                if col_name_answer in df.columns:
                    csv_val = row[col_name_answer]
                elif field in df.columns:
                    csv_val = row[field]

                if pd.isna(csv_val):
                    csv_val_str = "None mentioned"
                else:
                    csv_val_str = str(csv_val).strip()
                expected_output[field] = csv_val_str

            expected_json_str = json.dumps(expected_output, indent=2, ensure_ascii=False)

            jsonl_record = {
                "messages": [
                    {"role": "system", "content": format_system_prompt()},
                    {"role": "user", "content": (
                        "Extract the following information from the legal document text below.\n\n"
                        "Fields to extract:\n" +
                        "\n".join(f'"{fi}"' for fi in fields_list) +
                        f"\n\nDocument Text:\n{doc_text}"
                    )},
                    {"role": "assistant", "content": expected_json_str}
                ]
            }

            f.write(json.dumps(jsonl_record, ensure_ascii=False) + "\n")
            valid_documents += 1

    print(f"\n✅ Dataset generation complete!")
    print(f"   CSV rows     : {len(df)}")
    print(f"   Matched      : {valid_documents}")
    print(f"   Not matched  : {len(not_found)}")
    print(f"   File saved   : {os.path.abspath(output_path)}")
    if not_found:
        print(f"\nFiles not matched (first 15):")
        for name in not_found[:15]:
            print(f"  - {name}")

if __name__ == "__main__":
    create_finetuning_dataset()
