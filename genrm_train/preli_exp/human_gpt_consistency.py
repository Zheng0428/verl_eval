import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, matthews_corrcoef, confusion_matrix

def calculate_consistency(excel_file, sheet_name="Human Annotation", 
                         gpt_col="gpt4o_score", human_col="Weihao Annotation",
                         task_filter=None):
    """
    Calculate consistency metrics between human and GPT annotations.
    
    Args:
        excel_file (str): Path to the Excel file
        sheet_name (str): Name of the sheet containing annotations
        gpt_col (str): Column name for GPT annotations
        human_col (str): Column name for human annotations
        task_filter (str, optional): If provided, calculate metrics only for this task
        
    Returns:
        dict: Dictionary containing various consistency metrics
    """
    # Read the excel file
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    
    # Apply task filter if provided
    if task_filter is not None:
        df = df[df['task'] == task_filter]
    
    # Ensure annotations are binary (0 or 1)
    df[gpt_col] = df[gpt_col].astype(int)
    df[human_col] = df[human_col].astype(int)
    
    # Extract annotations
    gpt_annotations = df[gpt_col].values
    human_annotations = df[human_col].values
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(human_annotations, gpt_annotations),
        "cohen_kappa": cohen_kappa_score(human_annotations, gpt_annotations),
        "f1_score": f1_score(human_annotations, gpt_annotations),
        "matthews_correlation": matthews_corrcoef(human_annotations, gpt_annotations)
    }
    
    # Generate confusion matrix
    tn, fp, fn, tp = confusion_matrix(human_annotations, gpt_annotations).ravel()
    metrics["true_positives"] = tp
    metrics["true_negatives"] = tn
    metrics["false_positives"] = fp
    metrics["false_negatives"] = fn
    
    # Calculate agreement rates per category
    pos_agreement = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    neg_agreement = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0
    
    metrics["positive_agreement"] = pos_agreement
    metrics["negative_agreement"] = neg_agreement
    
    return metrics

def calculate_metrics_by_task(excel_file, sheet_name="Human Annotation", 
                             gpt_col="gpt4o_score", human_col="Weihao Annotation",
                             output_csv=None):
    """
    Calculate consistency metrics for each task and overall.
    
    Args:
        excel_file (str): Path to the Excel file
        sheet_name (str): Name of the sheet containing annotations
        gpt_col (str): Column name for GPT annotations
        human_col (str): Column name for human annotations
        output_csv (str, optional): Path to output CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing metrics for each task
    """
    # Read the excel file to get all tasks
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    
    if 'task' not in df.columns:
        raise ValueError("The Excel file does not contain a 'task' column")
    
    # Get unique tasks
    tasks = df['task'].unique()
    
    # Calculate metrics for each task
    results = []
    for task in tasks:
        metrics = calculate_consistency(excel_file, sheet_name, gpt_col, human_col, task_filter=task)
        metrics['task'] = task
        results.append(metrics)
    
    # Calculate overall metrics
    overall_metrics = calculate_consistency(excel_file, sheet_name, gpt_col, human_col)
    overall_metrics['task'] = 'Overall'
    results.append(overall_metrics)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Reorder columns to put task first
    cols = ['task'] + [col for col in results_df.columns if col != 'task']
    results_df = results_df[cols]
    
    # Output to CSV if requested
    if output_csv:
        results_df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
    
    return results_df

def print_metrics(metrics):
    """Print formatted metrics"""
    print("\n=== Human-GPT Annotation Consistency ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Matthews Correlation: {metrics['matthews_correlation']:.4f}")
    
    print("\n=== Confusion Matrix ===")
    print(f"True Positives: {metrics['true_positives']}")
    print(f"True Negatives: {metrics['true_negatives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    
    print("\n=== Agreement Rates ===")
    print(f"Positive Agreement: {metrics['positive_agreement']:.4f}")
    print(f"Negative Agreement: {metrics['negative_agreement']:.4f}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Calculate human-GPT annotation consistency")
    parser.add_argument("--file", type=str,  help="Path to Excel file", default="/Users/bytedance/Desktop/verl/Results for genrm (2).xlsx")
    parser.add_argument("--sheet", type=str, default="Human Annotation new", help="Sheet name with annotations")
    parser.add_argument("--gpt_col", type=str, default="gpt_correct", help="Column name for GPT annotations")
    parser.add_argument("--human_col", type=str, default="Weihao Ann", help="Column name for human annotations")
    parser.add_argument("--by_task", action="store_true", help="Calculate metrics for each task separately")
    parser.add_argument("--output_csv", type=str, help="Path to output CSV file")
    
    args = parser.parse_args()
    
    if args.by_task or args.output_csv:
        results_df = calculate_metrics_by_task(
            args.file, 
            sheet_name=args.sheet, 
            gpt_col=args.gpt_col, 
            human_col=args.human_col,
            output_csv=args.output_csv
        )
        print("\nResults by task:")
        print(results_df.to_string())
    else:
        metrics = calculate_consistency(
            args.file, 
            sheet_name=args.sheet, 
            gpt_col=args.gpt_col, 
            human_col=args.human_col
        )
        print_metrics(metrics)
    
if __name__ == "__main__":
    main()
