import json
import os
import csv
import random
import pandas as pd
from collections import defaultdict

# Define paths
base_dir = "/Users/bytedance/Desktop/verl/genrm_train/preli_exp/rule-based-eval"
output_dir = "/Users/bytedance/Desktop/verl/genrm_train/preli_exp/human_annotation"
os.makedirs(output_dir, exist_ok=True)
tasks = ["deepscaler", "skywork", "math", "orz"]
output_csv = os.path.join(output_dir, "human_annotation_samples.csv")

def load_jsonl(file_path):
    data_list = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            data_list.append(data)
    return data_list

# Function to extract examples for annotation
def sample_examples(results, n=50):
    # If there are fewer than n examples, return all of them
    if len(results) <= n:
        return results
    
    # Otherwise, randomly sample n examples
    return random.sample(results, n)

# Store all sampled examples
all_samples = []

# Process each task
for task in tasks:
    results_path = os.path.join(base_dir, task, "results.jsonl")
    
    if not os.path.exists(results_path):
        print(f"Warning: Results file for {task} not found at {results_path}")
        continue
    
    print(f"Processing {task}...")
    results = load_jsonl(results_path)
    
    # Sample examples
    sampled_results = sample_examples(results, 50)
    
    # Extract required fields for each sample
    for result in sampled_results:
        question = result.get('question', '')
        gt = result.get('answer', '')
        
        # Each result may have multiple predictions
        # For simplicity, just take the first prediction and its corresponding judgment
        preds = result.get('pred', [])
        pred = preds[0] if preds else ''
        
        gpt_correct_list = result.get('gpt_correct', [])
        gpt_correct = gpt_correct_list[0] if gpt_correct_list else None
        
        # It's not clear from the files what 'gpt_judge' refers to,
        # but we'll include it if it exists
        gpt_judge = result.get('gpt_judge', [''])[0] if isinstance(result.get('gpt_judge', ''), list) else result.get('gpt_judge', '')
        
        all_samples.append({
            'task': task,
            'question': question,
            'gt': gt,
            'pred': pred,
            'gpt_judge': gpt_judge,
            'gpt_correct': 1 if gpt_correct else 0,
        })

# Write to CSV
with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['task', 'question', 'gt', 'pred', 'gpt_judge', 'gpt_correct']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_samples)

print(f"Created human annotation file with {len(all_samples)} samples at {output_csv}")

# Also create separate CSV files for each task
task_samples = defaultdict(list)
for sample in all_samples:
    task_samples[sample['task']].append(sample)

for task, samples in task_samples.items():
    task_csv = os.path.join(output_dir, f"human_annotation_{task}.csv")
    with open(task_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['task', 'question', 'gt', 'pred', 'gpt_judge', 'gpt_correct']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)
    print(f"Created task-specific annotation file with {len(samples)} samples at {task_csv}")
