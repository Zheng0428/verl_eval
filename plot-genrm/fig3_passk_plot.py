import json 
import numpy as np 
import matplotlib.pyplot as plt
import os
import pandas as pd
from typing import Union, List

def estimate_pass_at_k(
        num_samples: Union[int, List[int], np.ndarray],
        num_correct: Union[List[int], np.ndarray],
        k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = np.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = num_samples

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def calculate_pass_at_k(data, k_values):
    score_mat = []
    for sample in data:
        score_mat.append(sample['score'])
    
    max_len = max([len(s) for s in score_mat])
    
    # Pad shorter arrays with the last value
    for i, s in enumerate(score_mat):
        if len(s) < max_len:
            score_mat[i] = s + [s[-1]] * (max_len - len(s))
    
    # Convert score matrix to numpy array for easier manipulation
    score_mat_np = np.array(score_mat)
    
    # Calculate number of correct answers per problem
    num_correct = np.sum(score_mat_np, axis=1)
    
    pass_at_k = {}
    for k in k_values:
        if k <= max_len:  # Only calculate for k values that make sense
            pass_at_k_estimates = estimate_pass_at_k(max_len, num_correct, k)
            pass_at_k[k] = float(np.round(np.mean(pass_at_k_estimates) * 100, decimals=1))
    
    return pass_at_k

def main():
    # Define k values to calculate pass@k
    k_values = [1, 2, 3, 4, 5, 8, 10, 16, 20, 32, 40, 50, 64, 100, 128, 200, 256, 300,350, 400, 450, 512, 600, 700,  800, 900,  1024]
    base_dir = "/Users/bytedance/Desktop/plot-genrm/passk_jsonl"
    csv_results_path = "pass_at_k_results.csv"
    
    # Define file paths
    jsonl_files = {
        'Base': os.path.join(base_dir, 'qwen-7b-base.jsonl'),
        'Rule-Based': os.path.join(base_dir, 'rule-base-step480-deepscaler.jsonl'),
        'R1-Qwen-1.5B': os.path.join(base_dir, 'r1-1.5b-as-judge-deepscaler.jsonl')
    }
    legend_name= {
        'Base': 'Base',
        'Rule-Based': 'HF Verifier',
        'R1-Qwen-1.5B': 'HF + R1-Qwen-1.5B'
    }
    # Check if results CSV already exists
    if os.path.exists(csv_results_path):
        print(f"Loading pass@k values from {csv_results_path}")
        # Load from CSV
        df_results = pd.read_csv(csv_results_path)
        
        # Convert to the format we need for plotting
        df_data = {'K': df_results['K'].tolist()}
        for model in jsonl_files.keys():
            if model in df_results.columns:
                df_data[model] = df_results[model].tolist()
    else:
        print("Calculating pass@k values...")
        # Calculate pass@k for each file
        results = {}
        for model, file_path in jsonl_files.items():
            print(f"Processing {model}...")
            data = load_jsonl(file_path)
            pass_at_k = calculate_pass_at_k(data, k_values)
            results[model] = pass_at_k
        
        # Prepare data for plotting and CSV
        df_data = {
            'K': []
        }
        
        for model in results:
            df_data[model] = []
        
        # Filter k values that exist in all models
        common_k = set.intersection(*[set(results[model].keys()) for model in results])
        common_k = sorted(list(common_k))
        
        for k in common_k:
            df_data['K'].append(k)
            for model in results:
                df_data[model].append(results[model][k])
        
        # Save to CSV for future use
        pd.DataFrame(df_data).to_csv(csv_results_path, index=False)
        print(f"Saved pass@k results to {csv_results_path}")
    
    # Create figure
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()

    # Colors for the different models
    colors = {
        'Base': '#3066BE',  # blue
        'Rule-Based': '#44087D',  # purple
        'R1-Qwen-1.5B': '#FF5A5F'  # red
    }

    markers = {
        'Base': 's',
        'Rule-Based': 'o',
        'R1-Qwen-1.5B': '^'
    }

    # Plot the lines
    for model in jsonl_files.keys():
        if model in df_data:
            ax.plot(df_data['K'], df_data[model], color=colors[model], 
                    label=model, marker=markers[model], linestyle='-', 
                    linewidth=4, markersize=9)

    # Set log scale for x-axis since K increases exponentially
    ax.set_xscale('log', base=2)
    
    # Set x-ticks to be powers of 2 from 1 to 1024
    power_of_2_ticks = [2**i for i in range(11)]  # 2^0 to 2^10 (1 to 1024)
    ax.set_xticks(power_of_2_ticks)
    ax.set_xticklabels([str(int(k)) for k in power_of_2_ticks], fontsize=20)
    ax.tick_params(axis='x', labelsize=20)
    # Increase y-axis tick label font size
    ax.tick_params(axis='y', labelsize=20)

    # # Add labels for each point
    # for model in jsonl_files.keys():
    #     if model in df_data:
    #         for i, (k, val) in enumerate(zip(df_data['K'], df_data[model])):
    #             ax.annotate(f'{val:.1f}', 
    #                         (k, val),
    #                         xytext=(0, 7),
    #                         textcoords='offset points',
    #                         ha='center',
    #                         va='bottom',
    #                         color=colors[model],
    #                         fontsize=14)

    # Set title and labels
    ax.set_xlabel("K", fontsize=22)
    ax.set_ylabel("Pass@K Accuracy (%)", fontsize=22)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Set y-axis limits with some padding
    model_max_values = [max(df_data[model]) for model in df_data if model != 'K']
    if model_max_values:
        y_max = max(model_max_values) * 1.1
        ax.set_ylim(0, y_max)

    # Add legend
    ax.legend(fontsize=22, loc='best', labels=legend_name.values())

    # Create Figures directory if it doesn't exist
    os.makedirs("Figures", exist_ok=True)
    
    # Save and show the figure
    plt.tight_layout()
    plt.savefig(os.path.join("Figures", "fig3_pass_at_k_comparison.pdf"), bbox_inches='tight', dpi=500)
    plt.savefig(os.path.join("Figures", "fig3_pass_at_k_comparison.png"), bbox_inches='tight', dpi=500)
    plt.show()

if __name__ == "__main__":
    main()