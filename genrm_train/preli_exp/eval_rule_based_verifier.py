from math_verify import parse, verify
import json
import os
import csv
import numpy as np
from verl.utils.reward_score.math import compute_score_without_extraction as verl_compute_score



model_config = {
    "DeepSeek-R1-Distill-Qwen-7B-new" : "size_1000_val_deepseek-r1_-1_seed0_t1.0_s0_e-1_generate_response_-1_reorg.jsonl",
    "DeepSeek-R1-Distill-Qwen-32B-new" : "size_1000_val_deepseek-r1_-1_seed0_t1.0_s0_e-1_generate_response_-1_reorg.jsonl",
    "Qwen2.5-32B-Instruct-qwen-boxed": "size_1000_val_qwen-boxed_-1_seed0_t1.0_s0_e-1_generate_response_-1_reorg.jsonl",
    "Qwen2.5-Math-7B-Instruct-qwen-boxed": "size_1000_val_qwen-boxed_-1_seed0_t1.0_s0_e-1_generate_response_-1_reorg.jsonl",
}

model_name = "Qwen2.5-Math-7B-Instruct-qwen-boxed"
Base_dir = f"/mnt/bn/tiktok-mm-5/aiic/users/huangyuzhen/verl/genrm_train/preli_exp/reorg_gpt_labeled_data_new/{model_name}"
task_file={
    "deepscaler": f"{Base_dir}/rollout_deepscaler/{model_config[model_name]}",
    "skywork": f"{Base_dir}/rollout_skywork/{model_config[model_name]}",
    "math": f"{Base_dir}/rollout_math/{model_config[model_name]}",
    "orz": f"{Base_dir}/rollout_orz/{model_config[model_name]}",
}

save_dir = f"/mnt/bn/tiktok-mm-5/aiic/users/huangyuzhen/verl/genrm_train/preli_exp/rule-based-eval/{model_name}"

def save_jsonl(data_list, file_path):
    with open(file_path, 'w') as f:
        for data in data_list:
            f.write(json.dumps(data) + '\n')

def load_jsonl(file_path):
    data_list = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            data_list.append(data)
    return data_list

def calculate_metrics(predictions, ground_truth):
    """
    Calculate precision, recall, F1, and accuracy
    
    Args:
        predictions: List of binary predictions (1 for correct, 0 for incorrect)
        ground_truth: List of binary ground truth (1 for correct, 0 for incorrect)
    
    Returns:
        Dictionary with precision, recall, F1, and accuracy
    """
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    TP = np.sum((predictions == 1) & (ground_truth == 1))
    FP = np.sum((predictions == 1) & (ground_truth == 0))
    TN = np.sum((predictions == 0) & (ground_truth == 0))
    FN = np.sum((predictions == 0) & (ground_truth == 1))
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }

# Dictionary to store all results for final summary
all_results = {}

for task, file_path in task_file.items():
    print(f"Processing task: {task}")
    # Create save directory if it doesn't exist
    os.makedirs(f"{save_dir}/{task}", exist_ok=True)
    
    data_list = load_jsonl(file_path)
    
    # Lists to store binary predictions and ground truth
    hf_binary_preds = []
    verl_binary_preds = []
    qwen_binary_preds = []
    ground_truth = []
    
    for data in data_list:
        gt = data['answer']
        if "\\boxed" not in gt:
            hf_gt = f"\\boxed{{{gt}}}"
        else:
            gt = hf_gt
        hf_verify_results = []
        verl_verify_results = []
        
        # Extract ground truth correctness
        gt_correct_list = data.get('gpt_correct', [])
        
        for i, pred in enumerate(data['pred']):
            # Get ground truth for this prediction
            if i < len(gt_correct_list):
                gt_correct = gt_correct_list[i]
                ground_truth.append(1 if gt_correct else 0)
            else:
                # Skip if no ground truth available
                continue
            # Process prediction
            if pred is None:
                pred = ''
            if "\\boxed" not in pred:
                hf_pred = f"\\boxed{{{pred}}}"
            else:
                hf_pred=pred
            
            # HF verification

            hf_score = verify(gold=parse(hf_gt), target=parse(hf_pred))
            hf_verify_results.append(hf_score)
            hf_binary_preds.append(1 if hf_score > 0 else 0)
            
            # VERL verification
            verl_score = verl_compute_score(solution_str=pred, ground_truth=gt)
            verl_verify_results.append(verl_score)
            verl_binary_preds.append(1 if verl_score > 0 else 0)
            
            # Qwen verification
            if 'score' in data and i < len(data['score']):
                qwen_score = data['score'][i]
                qwen_binary_preds.append(1 if qwen_score > 0 else 0)
        
        # Store verification results in data
        data['hf_verify_results'] = hf_verify_results
        data['verl_verify_results'] = verl_verify_results
        data['qwen_verify_results'] = data.get('score', [])
    
    # Save detailed results as JSONL
    save_jsonl(data_list, f"{save_dir}/{task}/results.jsonl")
    
    # Calculate metrics
    hf_metrics = calculate_metrics(hf_binary_preds, ground_truth)
    verl_metrics = calculate_metrics(verl_binary_preds, ground_truth)
    qwen_metrics = calculate_metrics(qwen_binary_preds, ground_truth)
    
    # Store results for the final summary
    all_results[task] = {
        "HuggingFace": hf_metrics,
        "VERL": verl_metrics,
        "Qwen": qwen_metrics,
        "samples": len(ground_truth)
    }
    
    # Create results table for this task
    results_table = [
        ["Verifier", "Precision", "Recall", "F1", "Accuracy"],
        ["HuggingFace", hf_metrics["precision"], hf_metrics["recall"], hf_metrics["f1"], hf_metrics["accuracy"]],
        ["VERL", verl_metrics["precision"], verl_metrics["recall"], verl_metrics["f1"], verl_metrics["accuracy"]],
        ["Qwen", qwen_metrics["precision"], qwen_metrics["recall"], qwen_metrics["f1"], qwen_metrics["accuracy"]]
    ]
    
    # Save results as CSV for this task
    with open(f"{save_dir}/{task}/results.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in results_table:
            writer.writerow(row)
    
    print(f"Results for {task}:")
    print(f"HuggingFace: Precision={hf_metrics['precision']:.4f}, Recall={hf_metrics['recall']:.4f}, F1={hf_metrics['f1']:.4f}, Accuracy={hf_metrics['accuracy']:.4f}")
    print(f"VERL: Precision={verl_metrics['precision']:.4f}, Recall={verl_metrics['recall']:.4f}, F1={verl_metrics['f1']:.4f}, Accuracy={verl_metrics['accuracy']:.4f}")
    print(f"Qwen: Precision={qwen_metrics['precision']:.4f}, Recall={qwen_metrics['recall']:.4f}, F1={qwen_metrics['f1']:.4f}, Accuracy={qwen_metrics['accuracy']:.4f}")
    print()

# Create a summary table with results for all tasks (original format)
summary_table = [
    ["Task", "Verifier", "Samples", "Precision", "Recall", "F1", "Accuracy"]
]

# Calculate weighted averages across all tasks
total_samples = sum(info["samples"] for info in all_results.values())
weighted_metrics = {
    "HuggingFace": {"precision": 0, "recall": 0, "f1": 0, "accuracy": 0},
    "VERL": {"precision": 0, "recall": 0, "f1": 0, "accuracy": 0},
    "Qwen": {"precision": 0, "recall": 0, "f1": 0, "accuracy": 0},
}

# Fill in the summary table and calculate weighted averages
for task, info in all_results.items():
    task_weight = info["samples"] / total_samples
    
    for verifier in ["HuggingFace", "VERL", "Qwen"]:
        # Add task results to the summary table
        summary_table.append([
            task, 
            verifier, 
            info["samples"],
            info[verifier]["precision"], 
            info[verifier]["recall"], 
            info[verifier]["f1"], 
            info[verifier]["accuracy"]
        ])
        
        # Update weighted averages
        for metric in ["precision", "recall", "f1", "accuracy"]:
            weighted_metrics[verifier][metric] += info[verifier][metric] * task_weight

# Add weighted average results to the summary table
for verifier in ["HuggingFace", "VERL", "Qwen"]:
    summary_table.append([
        "Average", 
        verifier, 
        total_samples,
        weighted_metrics[verifier]["precision"], 
        weighted_metrics[verifier]["recall"], 
        weighted_metrics[verifier]["f1"], 
        weighted_metrics[verifier]["accuracy"]
    ])

# Save the summary table (original format)
with open(f"{save_dir}/summary_results.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in summary_table:
        writer.writerow(row)

# Create a more readable structured table format
# This format has tasks as rows and metrics as columns, organized by verifier
structured_table = []

# Create the header row
header = ["Task", "Samples"]
for verifier in ["HuggingFace", "VERL", "Qwen"]:
    for metric in ["Precision", "Recall", "F1", "Accuracy"]:
        header.append(f"{verifier}_{metric}")
structured_table.append(header)

# Add task rows
tasks = sorted(all_results.keys())
for task in tasks:
    info = all_results[task]
    row = [task, info["samples"]]
    for verifier in ["HuggingFace", "VERL", "Qwen"]:
        for metric in ["precision", "recall", "f1", "accuracy"]:
            row.append(info[verifier][metric])
    structured_table.append(row)

# Add average row
avg_row = ["Average", total_samples]
for verifier in ["HuggingFace", "VERL", "Qwen"]:
    for metric in ["precision", "recall", "f1", "accuracy"]:
        avg_row.append(weighted_metrics[verifier][metric])
structured_table.append(avg_row)

# Save the structured summary table
with open(f"{save_dir}/structured_summary_results.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in structured_table:
        writer.writerow(row)

print("Structured summary results saved to", f"{save_dir}/structured_summary_results.csv")

# Print the summary table in a more readable format
print("\nSummary Results Table:")
print(f"{'Task':<12} {'Verifier':<12} {'Samples':<8} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Accuracy':<10}")
print("-" * 80)

for row in summary_table[1:]:
    task, verifier, samples, precision, recall, f1, accuracy = row
    print(f"{task:<12} {verifier:<12} {samples:<8} {precision:.4f}{' '*6} {recall:.4f}{' '*6} {f1:.4f}{' '*6} {accuracy:.4f}")


