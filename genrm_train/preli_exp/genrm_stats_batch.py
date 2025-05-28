import os
import csv
import argparse
import pandas as pd
import itertools
import subprocess
import random
import numpy as np
from genrm_stats_fnr import calculate_metrics, load_jsonl
from genrm_config import * 
import re

def extract_task_from_path(file_path):
    """Extract task name from file path."""
    # Look for task names like genrm_skywork, genrm_orz, etc.
    task_names = ["genrm_skywork", "genrm_orz", "genrm_deepscaler", "genrm_math"]
    for task_name in task_names:
        if task_name in file_path:
            return task_name
    
    # If not found in path, try to extract from directory structure
    try:
        # Assuming paths like /path/to/model/taskname/filename.jsonl
        task_dir = os.path.basename(os.path.dirname(file_path))
        if task_dir.startswith("genrm_"):
            return task_dir
    except:
        pass
    
    return "unknown"

def calculate_random_baseline(gt_list, org_example_num, current_example_num):
    """
    Calculate metrics for random guessing based on actual ground truth distribution.
    This simulates a random classifier that makes predictions with probabilities 
    matching the class distribution in the data.
    """
    # Get proportion of positive examples in ground truth
    pos_ratio = sum(gt_list) / len(gt_list) if gt_list else 0.5
    
    # Generate random predictions with the same class distribution
    np.random.seed(42)  # For reproducibility
    random_preds = []
    
    # Method 1: Random sampling based on true distribution
    random_preds = np.random.random(len(gt_list)) < 0.5
    random_preds = [float(x) for x in random_preds]
    
    # Calculate metrics using the same function as real models
    metrics = calculate_metrics(random_preds, gt_list, org_example_num, current_example_num)
    
    return metrics

def extract_score(pred, code, model_type):
    if model_type == "xverify":
        if code is not None:
            if code.strip() == '1' or code.strip().lower() == 'true' or code.strip().lower() == 'correct':
                score = 1.0
            elif code.strip() == '0' or code.strip().lower() == 'false' or code.strip().lower() == 'incorrect':
                score = 0.0
            else:
                score = 0.0
        else:
            score = 0.0
    elif model_type == "wenhu":
        ext_re = r"Final Decision:\s*(yes|no|true|false)"
        match = re.search(ext_re, code, re.IGNORECASE)
        if match:
            extracted_answer = match.group(1).strip().lower()
            if extracted_answer.lower() in ["yes", "true"]:
                score = 1.0
            elif extracted_answer.lower() in ["no", "false"]:
                score = 0.0
        else:
            score = 0.0
    else:
        if pred is not None:
            if pred.strip() == '1':
                score = 1.0
            elif pred.strip() == '0':
                score = 0.0
            else:
                score = 0.0
        else:
            score = 0.0
    return score

def run_stats_batch(args):
    """Run statistics for all models and tasks defined in config."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize results storage
    results = {
        "model": [],
        "task": [],
        "accuracy": [],
        "recall": [],
        "precision": [],
        "f1": [],
        "system_fnr": [],
        "system_precision": [],
        "system_recall": [],
        "system_f1": []
    }
    
    # Track statistics
    processed_count = 0
    missing_count = 0
    error_count = 0
    
    # For random baseline calculation
    random_baselines = {}
    
    # Generate all combinations of models and tasks
    tasks = list(TASK_ORG_FILES.keys())
    
    # Process each model-task combination
    for model_name, filename_template in MODEL_TEST_FILES.items():
        filename_template = filename_template.format(suffix=args.suffix)
        for task_name in tasks:
            # Construct the full path for the test file
            test_file = os.path.join(
                args.input_dir,
                model_name, 
                task_name, 
                filename_template
            )
            model_type = MODEL_TO_TYPE_MAP.get(model_name, "unknown")
            
            # Check if the file exists before processing
            if not os.path.exists(test_file):
                print(f"Missing file: {test_file}")
                missing_count += 1
                continue
                
            # Get corresponding original file
            org_file = TASK_ORG_FILES[task_name].format(suffix=args.suffix)
            if not os.path.exists(org_file):
                print(f"Missing original file: {org_file}")
                missing_count += 1
                continue
                
            # Get original example count
            org_example_num = TASK_EXAMPLE_COUNTS.get(task_name, 1000000000)
            print(f"org_example_num: {org_example_num}")
            # Load data
            try:
                rule_plus_genrm = []
                gt_list = []
                org_data = list(load_jsonl(org_file))
                test_data = list(load_jsonl(test_file))
                
                current_example_num = len(test_data)
                
                # Extract predictions and ground truth
                for org, test in zip(org_data, test_data):
                    assert org['idx'] == test['idx']
                    rule_based_judge = org['ori_rollout_info']['rule_based_score']
                    gpt_judge = org['answer']
                    
                    pred = test['pred'][0]
                    code = test['code'][0]
                    score = extract_score(pred, code, model_type)
                    
                    # if rule_based_judge:
                    #     score = 1.0
                    rule_plus_genrm.append(score)
                    gt_list.append(gpt_judge)
                
                # Calculate random baseline if we haven't for this task yet
                if task_name not in random_baselines:
                    random_metrics = calculate_random_baseline(gt_list, org_example_num, current_example_num)
                    if random_metrics:
                        random_baselines[task_name] = random_metrics
                
                # Calculate metrics
                metrics = calculate_metrics(rule_plus_genrm, gt_list, org_example_num, current_example_num)
                
                if metrics:
                    acc, rec, f1, cm, system_fnr, system_precision, system_recall, system_f1 = metrics
                    
                    # Store results
                    results["model"].append(model_name)
                    results["task"].append(task_name)
                    results["accuracy"].append(acc)
                    results["recall"].append(rec)
                    results["f1"].append(f1)
                    results["system_fnr"].append(system_fnr)
                    results["system_precision"].append(system_precision)
                    results["system_recall"].append(system_recall)
                    results["system_f1"].append(system_f1)
                    
                    # Calculate precision from confusion matrix
                    if (cm[0][1] + cm[1][1]) > 0:
                        precision = cm[1][1] / (cm[0][1] + cm[1][1])
                    else:
                        precision = 0
                    results["precision"].append(precision)
                    
                    processed_count += 1
                
            except Exception as e:
                print(f"Error processing {model_name} on {task_name}: {str(e)}")
                error_count += 1
    
    # Add random baseline results
    for task_name, random_metrics in random_baselines.items():
        if random_metrics:
            acc, rec, f1, cm, system_fnr, system_precision, system_recall, system_f1 = random_metrics
            
            # Calculate precision from confusion matrix
            if (cm[0][1] + cm[1][1]) > 0:
                precision = cm[1][1] / (cm[0][1] + cm[1][1])
            else:
                precision = 0
                
            # Add as first entry
            results["model"].insert(0, "Random-Baseline")
            results["task"].insert(0, task_name)
            results["accuracy"].insert(0, acc)
            results["recall"].insert(0, rec)
            results["f1"].insert(0, f1)
            results["system_fnr"].insert(0, system_fnr)
            results["system_precision"].insert(0, system_precision)
            results["system_recall"].insert(0, system_recall)
            results["system_f1"].insert(0, system_f1)
            results["precision"].insert(0, precision)
    
    # Save results to CSV files if we have any results
    if results["model"]:
        df = pd.DataFrame(results)
        
        # Save all results to one file
        all_results_path = os.path.join(args.output_dir, "all_metrics.csv")
        df.to_csv(all_results_path, index=False)
        print(f"Saved: {all_results_path}")
        
        # Save individual metric files
        metrics = ["accuracy", "recall", "precision", "f1", "system_fnr", "system_precision", "system_recall", "system_f1"]
        for metric in metrics:
            metric_df = df[["model", "task", metric]]
            metric_path = os.path.join(args.output_dir, f"{metric}.csv")
            # metric_df.to_csv(metric_path, index=False)
            # print(f"Saved: {metric_path}")
        
        # Create pivot tables with tasks on the x-axis (columns) and models on the y-axis (rows)
        for metric in metrics:
            # Create pivot with models as rows and tasks as columns
            pivot_df = df.pivot(index="model", columns="task", values=metric)
            
            # Move Random-Baseline to the top
            if "Random-Baseline" in pivot_df.index:
                random_row = pivot_df.loc["Random-Baseline"]
                pivot_df = pivot_df.drop("Random-Baseline")
                pivot_df = pd.concat([pd.DataFrame([random_row], index=["Random-Baseline"]), pivot_df])
            
            pivot_path = os.path.join(args.output_dir, f"{metric}_comparison.csv")
            pivot_df.to_csv(pivot_path)
            print(f"Saved: {pivot_path}")
            
            # Also save a transposed version for reference
            # pivot_df_transposed = pivot_df.transpose()
            # transpose_path = os.path.join(args.output_dir, f"{metric}_comparison_transposed.csv")
            # pivot_df_transposed.to_csv(transpose_path)
            # print(f"Saved: {transpose_path}")
            
        # Print summary
        print(f"\nSummary: Processed {processed_count} model-task combinations")
        if missing_count > 0:
            print(f"         Missing files: {missing_count}")
        if error_count > 0:
            print(f"         Processing errors: {error_count}")
    else:
        print("No results generated. Check file paths and configurations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/Users/bytedance/Desktop/verl/genrm_train/preli_exp/genrm_eval",
                        help="Input directory containing model test files")
    parser.add_argument("--output_dir", type=str, default="./genrm_results", 
                        help="Directory to save output CSV files")
    parser.add_argument("--suffix", type=str, default="false_only",
                        help="Suffix for the input files")
    args = parser.parse_args()
    
    # Run batch statistics
    run_stats_batch(args) 