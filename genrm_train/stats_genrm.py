import json
import numpy as np
import os
import glob
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

def calculate_metrics(jsonl_file):
    # Lists to store predictions and ground truth
    predictions = []
    ground_truth = []
    
    # Read the jsonl file
    with open(jsonl_file, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                try:
                    data = json.loads(line)
                    # Extract prediction and ground truth
                    pred = data.get('pred')
                    gt = data.get('answer')
                    
                    if pred is not None and gt is not None:
                        # Convert prediction string to int
                        if isinstance(pred, list) and len(pred) > 0:
                            # Convert to binary (0 or 1)
                            pred_value = 1 if pred[0] == '1' else 0
                        else:
                            continue
                            
                        # Convert ground truth to binary (0 or 1)
                        gt_value = 1 if float(gt) == 1.0 else 0
                        
                        predictions.append(pred_value)
                        ground_truth.append(gt_value)
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error processing line: {e}")
    
    # Calculate metrics
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

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
        
        # Optional second argument to control recursion (default is recursive)
        recursive = True
        if len(sys.argv) > 2:
            recursive = sys.argv[2].lower() in ['true', 't', 'yes', 'y', '1']
        
        if os.path.isdir(path):
            # If input is a directory, process all jsonl files in it
            process_folder(path, recursive)
        elif os.path.isfile(path) and path.endswith('.jsonl'):
            # If input is a single jsonl file, process it directly
            print(f"Calculating metrics for: {path}")
            calculate_metrics(path)
        else:
            print(f"Error: {path} is not a valid JSONL file or directory.")
    else:
        print("Usage: python stats_genrm.py <path> [recursive=True/False]")
        print("  <path>: Path to a JSONL file or directory containing JSONL files")
        print("  [recursive]: Optional. Set to False to disable recursive search in subdirectories")
