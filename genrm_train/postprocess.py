import json
import argparse
from pathlib import Path
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_boxed_value(text):
    """Extract value from \\boxed{X} format."""
    if not text:
        return None
    
    match = re.search(r'\\boxed\{([^}]*)\}', text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

def process_data(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    processed_data = []
    skipped_count = 0
    MAX_NUM_PER_INPUT = 4
    for item in data:
        ref_score = item['extra_info']['4o Reward Score']
        skip = True
        num_valid_completions = 0  # Filter completions by setting text to empty if scores don't match
        for comp in item["completion"]:
            genrm_score_raw = comp['GenRM reward_score']
            if genrm_score_raw == "":
                comp["text"] = ""
                continue
            boxed_value = extract_boxed_value(genrm_score_raw)
        
            if  boxed_value != ref_score:
                comp["text"] = ""
                continue
            if num_valid_completions >= MAX_NUM_PER_INPUT:
                continue
            new_item = {
                "input": item["genrm_prompt"],
                "target": comp["text"],
                "extra_info": item["extra_info"]
            }
            skip = False
            processed_data.append(new_item)
            num_valid_completions += 1
        if skip:
            skipped_count += 1
    
    # Write processed data
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=4)
    with open(output_file.replace(".json", "_mid.json"), 'w') as f:
        json.dump(data, f, indent=4)
        
        
    logging.info(f"Processed {len(data)} items, skipped {skipped_count}, output {len(processed_data)} items")
    return processed_data

def main():
    parser = argparse.ArgumentParser(description="Process GenRM training data")
    parser.add_argument("--input", type=str, default="genrm_train/data/deepscaler/generated_data.json", 
                        help="Path to input JSON file")
    parser.add_argument("--output", type=str, default="genrm_train/data/deepscaler/processed_data.json", 
                        help="Path to output JSON file")
    
    args = parser.parse_args()
    
    input_file = args.input
    output_file = args.output
    
    processed_data = process_data(input_file, output_file)
    logging.info(f"Data processing complete. Output file: {output_file}")

if __name__ == "__main__":
    main()
