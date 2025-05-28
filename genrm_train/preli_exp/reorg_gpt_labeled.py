import os
import json
import argparse
import re
from collections import defaultdict


def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def save_jsonl(data, file_path):
    with open(file_path, 'w') as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')

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
        
        # Additional patterns
        r'\\text\{\[Reward\s+Score\]\}\s*=\s*(\d+)',
        r'\\textrm\{\[Reward\s+Score\]\}\s*=\s*(\d+)',
        r'\\boxed\{\\text\{\"Reward\s+Score\"\s*=\s*(\d+)\}\}',
        r'\\\[\s*\\text\{\[Reward\s+Score\]\s*=\s*\}\s*(\d+)\s*\\\]',
        r'\\text\{Reward\s+Score\}\s*=\s*(\d+)',
        r'\\\[Reward\s+Score\s*=\s*(\d+)\\\]',
        r'\\\[\s*\\text\{Reward\s+Score\}\s*=\s*(\d+)\s*\\\]',
        r'\\\[\s*\\text\{\"Reward\s+Score\"\}\s*=\s*(\d+)\s*\\\]',
        r'\\\[\s*\\text\{\"Reward\s+Score\"\s*=\s*(\d+)\}\s*\\\]',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Reorganize GPT labeled data')
    parser.add_argument('--labeled_data_path', type=str, required=True, 
                        help='Path to the GPT labeled data file')
    parser.add_argument('--original_data_path', type=str, required=True, 
                        help='Path to the original data file')
    parser.add_argument('--output_path', type=str, required=True, 
                        help='Path to save the reorganized data')
    parser.add_argument('--group_by', type=str, default='question', choices=['question', 'idx'], 
                        help='Field to group the data by (question or idx)')
    args = parser.parse_args()
    
    data = load_jsonl(args.labeled_data_path)
    max_idx = max([item['idx'] for item in data])
    print(f"Max idx: {max_idx}")
    
    # Use defaultdict to simplify processing
    data_dict = defaultdict(dict)
    
    # Group items with the same group_by field
    group_by_field = args.group_by
    for item in data:
        pred_idx = item['pred_index']
        data_dict[item[group_by_field]][pred_idx] = item
    
    original_data = load_jsonl(args.original_data_path)
    original_data_dict = {item['idx']: item for item in original_data}
    
    output_data = []
    for idx, items in original_data_dict.items():
        items['gpt_judge'] = [""] * len(items['pred'])
        # items['reward_scores'] = [0] * len(items['pred'])
        items['gpt_correct'] = [True] * len(items['pred'])
        
        # The key to check in data_dict depends on the group_by field
        key_to_check = items['question'] if group_by_field == 'question' else idx
        
        for pred_idx, _ in enumerate(items['pred']):
            if key_to_check in data_dict and pred_idx in data_dict[key_to_check]:
                judge_text = data_dict[key_to_check][pred_idx]['generation']
                items['gpt_judge'][pred_idx] = judge_text
                
                # Extract reward score
                score = extract_reward_score(judge_text)
                # items['reward_scores'][pred_idx] = score if score is not None else 0
                
                # Determine if the response is correct (score > 0)
                items['gpt_correct'][pred_idx] = score is not None and score > 0
            else:
                items['gpt_judge'][pred_idx] = ""
                # items['gpt_scores'][pred_idx] = 1
                items['gpt_correct'][pred_idx] = True
                
        output_data.append(items)
    
    save_jsonl(output_data, args.output_path)
    print(f"Reorganized data saved to {args.output_path}")


if __name__ == "__main__":
    main()
