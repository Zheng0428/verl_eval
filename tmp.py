import os
import json
import re
input_dir = "/Users/bytedance/Desktop/verl/genrm_train/preli_exp/genrm_eval/tiger-verifier"

# Find all JSONL files that are two levels down from input_dir
file_list = []
for root, dirs, files in os.walk(input_dir):
    # Calculate the depth relative to input_dir
    rel_path = os.path.relpath(root, input_dir)
    depth = len(rel_path.split(os.sep)) if rel_path != '.' else 0
    
    # If we're at depth 2, look for jsonl files
    if depth == 1:
        for file in files:
            if file.endswith('.jsonl'):
                file_list.append(os.path.join(root, file))
# file_list = file_list[:1]
# Process each JSONL file
all_data = []
scores = []
pattern = r"Final Decision:\s*(yes|no|true|false)"

for file_path in file_list:
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Each line in a JSONL file is a valid JSON object
            data = json.loads(line.strip())
            all_data.append(data)
            
            # Extract decision from code field
            if "code" in data:
                match = re.search(pattern, data["code"][0].lower(), re.IGNORECASE)
                if match:
                    decision = match.group(1).strip().lower()
                    if decision.lower() in ["yes", "true"]:
                        scores.append(1)
                    elif decision.lower() in ["no", "false"]:
                        scores.append(0)
                    else:
                        print(f"Unexpected decision: {decision}")
                        print("="*100)
                        scores.append(None)
                else:
                    # print(f"No decision found in record: {data.get('id', 'unknown id')}")
                    print(data["code"][0])
                    print("="*100)
                    scores.append(None)
            else:
                print(f"No code field found in record: {data.get('id', 'unknown id')}")
                scores.append(None)

# Now all_data contains all the records from all JSONL files
print(f"Read {len(all_data)} records from {len(file_list)} JSONL files")
print(f"Found {len([s for s in scores if s is not None])} valid decisions")
print(f"Found {scores.count(1)} 'yes/true' decisions and {scores.count(0)} 'no/false' decisions") 
print(f'Found {scores.count(None)} None decisions')

