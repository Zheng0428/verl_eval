import json
import re

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def extract_reward_score(text):
    patterns = [
        # Original patterns with added whitespace flexibility
        r'\[\s*Reward\s+Score\s*\]\s*=\s*(\d+)',
        r'\\\[\s*Reward\s+Score\s*\\\]\s*=\s*(\d+)',
        r'\[\s*\"Reward\s+Score\"\s*=\s*(\d+)\s*\\*\]\'*',
        r'\\\[\s*\\text\{\[Reward\s+Score\]\}\s*=\s*(\d+)\s*\\\]',
        r'\\\[\s*\\text\{\\"Reward\s+Score\\"\s*=\s*(\d+)\}\s*\\\]',
        r'\\\[\s*\\text\{\\"Reward\s+Score\\"\}\s*=\s*(\d+)\s*\\\]',
        r'\\\[\s*\\\[\\text\{Reward\s+Score\}\\\]\s*=\s*(\d+)\s*\\\]',
        r"\\\[\s*\[\\text\\{Reward\s+Score\\}\]\s*=\s*(\d+)\\\]",
        r'\\\[\s*\[\s*\\text\s*\{\s*Reward\s+Score\s*\}\s*\]\s*=\s*(\d+)\s*\\\]',
        
        # New patterns
        r'\\text\{\[Reward\s+Score\]\}\s*=\s*(\d+)',
        r'\\textrm\{\[Reward\s+Score\]\}\s*=\s*(\d+)',
        r'\\boxed\{\\text\{\"Reward\s+Score\"\s*=\s*(\d+)\}\}',
        r'\\\[\s*\\text\{\[Reward\s+Score\]\s*=\s*\}\s*(\d+)\s*\\\]',
        
        # Additional new patterns
        r'\\text\{Reward\s+Score\}\s*=\s*(\d+)',
        r'\\\[Reward\s+Score\s*=\s*(\d+)\\\]',
        r'\\\[\s*\\text\{Reward\s+Score\}\s*=\s*(\d+)\s*\\\]',
        r'\\\[\s*\\text\{\"Reward\s+Score\"\}\s*=\s*(\d+)\s*\\\]',
        
        # Newly requested pattern
        r'\\\[\s*\\text\{\"Reward\s+Score\"\s*=\s*(\d+)\}\s*\\\]',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    
    return None

def main():
    data = read_json("/Users/bytedance/Desktop/verl/genrm_train/data/deepscaler_iter1.1/trnlog_0420_all_data.json")
    count = 0
    for item in data:
        # Extract the score from the generation text
        extracted_score = extract_reward_score(item['generation'])
        
        # Check for mismatches between extracted score and stored score
        if item['4o Reward Score'] == 0 and (extracted_score != 0 or extracted_score is None):
            count += 1
            print(f"Mismatch found! Stored score: {item['4o Reward Score']}, Extracted score: {extracted_score}")
            print(item['generation'])
            print("-" * 80)
        item["4o Reward Score"] = float(extracted_score if extracted_score is not None else 0)
    print(f"Total mismatches found: {count}")
    with open("/Users/bytedance/Desktop/verl/genrm_train/data/deepscaler_iter1.1/trnlog_0420_all_data_fixed.json", "w") as f:
        json.dump(data, f, indent=4)

    

    
if __name__ == "__main__":
    main()


