import json
import numpy as np
import os
import glob
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import argparse

def calculate_metrics(predictions, ground_truth):
    if predictions and ground_truth:
        acc = accuracy_score(ground_truth, predictions)
        rec = recall_score(ground_truth, predictions, zero_division=0)
        f1 = f1_score(ground_truth, predictions, zero_division=0)
        cm = confusion_matrix(ground_truth, predictions)
        
        # Calculate percentage values for confusion matrix
        total_samples = len(ground_truth)
        class_0_total = np.sum(cm[0])
        class_1_total = np.sum(cm[1])
        
        # Print results with both decimal and percentage
        
        print("\nConfusion Matrix (count):")
        print("    Predicted 0  Predicted 1   Total")
        print(f"Actual 0    {cm[0][0]}           {cm[0][1]}         {class_0_total}")
        print(f"Actual 1    {cm[1][0]}           {cm[1][1]}         {class_1_total}")
        print(f"Total       {cm[0][0]+cm[1][0]}           {cm[0][1]+cm[1][1]}         {total_samples}")
        
        print("\nConfusion Matrix (percentage):")
        print("                Predicted 0      Predicted 1")
        if class_0_total > 0:
            print(f"Actual 0        {cm[0][0]/class_0_total*100:.2f}%          {cm[0][1]/class_0_total*100:.2f}%")
        else:
            print("Actual 0        0.00%          0.00%")
            
        if class_1_total > 0:
            print(f"Actual 1        {cm[1][0]/class_1_total*100:.2f}%          {cm[1][1]/class_1_total*100:.2f}%")
        else:
            print("Actual 1        0.00%          0.00%")
        
        # Additional metrics
        if class_1_total > 0:
            precision = cm[1][1] / (cm[0][1] + cm[1][1]) if (cm[0][1] + cm[1][1]) > 0 else 0
            print(f"\nPrecision: {precision:.4f} ({precision*100:.2f}%)") #Precision = True Positives / (True Positives + False Positives)
        print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"Recall: {rec:.4f} ({rec*100:.2f}%)")
        print(f"F1 Score: {f1:.4f} ({f1*100:.2f}%)")
        return acc, rec, f1, cm
    else:
        print("No valid data found for calculating metrics.")
        return None

def process_folder(folder_path, recursive=True):
    """Process all JSONL files in the given folder.
    
    Args:
        folder_path: Path to the folder containing JSONL files
        recursive: If True, search recursively through subdirectories
    """
    # Find all jsonl files in the folder and subdirectories if recursive is True
    if recursive:
        jsonl_files = glob.glob(os.path.join(folder_path, "**/*.jsonl"), recursive=True)
    else:
        jsonl_files = glob.glob(os.path.join(folder_path, "*.jsonl"))
    
    if not jsonl_files:
        print(f"No JSONL files found in {folder_path}" + (" (including subdirectories)" if recursive else ""))
        return
    
    print(f"Found {len(jsonl_files)} JSONL files to process.")
    
    # Process each file
    for jsonl_file in jsonl_files:
        print(f"\n{'='*50}")
        print(f"Calculating metrics for: {jsonl_file}")
        print(f"{'='*50}")
        calculate_metrics(jsonl_file)

def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
                yield json.loads(line)

import sys 
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    
    args.add_argument("--test_file", type=str, default="/Users/bytedance/Desktop/hdfs/r1-distilled-qwen-1.5b/genrm_deepscaler/val-r1_qwen_32b-wo_q_deepseek-r1_-1_seed0_t0.6_s0_e-1.jsonl")
    args.add_argument("--org_file", type=str, default="/Users/bytedance/Desktop/verl/examples/simplelr_math_eval/data/genrm_deepscaler/val-r1_qwen_32b-wo_q.jsonl")
    args = args.parse_args()
    
    
    rule_plus_genrm =  []
    gt_list = []
    org_data = load_jsonl(args.org_file)
    test_data = load_jsonl(args.test_file)
    
    for org, test in zip(org_data, test_data):
        assert org['idx'] == test['idx']
        rule_based_judge = org['ori_rollout_info']['rule_based_score']
        gpt_judge = org['answer']
        
        pred = test['pred'][0]
        if pred is not  None:
            if pred.strip() == '1':
                new_pred = 1.0
            elif pred.strip() == '0':
                new_pred = 0.0
            else:
                new_pred = 0.0
        else:
            new_pred = 0.0
        if rule_based_judge:
            new_pred = 1.0
        rule_plus_genrm.append(new_pred)
        gt_list.append(gpt_judge)

    calculate_metrics(rule_plus_genrm, gt_list)
    