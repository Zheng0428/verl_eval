import json 
import re 
import numpy as np 
import itertools
import matplotlib.pyplot as plt
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
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

def load_json(file_path):
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
    # File paths for step 0 and step 700
    step0_path = "/Users/bytedance/Desktop/verl/step0/test_qwen-boxed_-1_seed0_t1.0_s0_e-1.jsonl"
    step700_path = "/Users/bytedance/Desktop/verl/step700/test_qwen-boxed_-1_seed0_t1.0_s0_e-1.jsonl"
    
    # Load data
    step0_data = load_json(step0_path)
    step700_data = load_json(step700_path)
    
    # Create more k values for a smoother curve
    # We'll include a range of values, not just powers of 2
    k_values = []
    # for k in range(1,1024,50):
        # k_values.append(k)
    k_values = [1, 2, 3, 4, 5, 8, 10, 16, 20, 32, 40, 50, 64, 100, 128, 200, 256, 300, 400, 512, 600, 700,800, 900,1000, 1024]
    
    # Calculate pass@k for both datasets
    step0_results = calculate_pass_at_k(step0_data, k_values)
    step700_results = calculate_pass_at_k(step700_data, k_values)
    
    # Sort k values for plotting
    sorted_k_values = sorted(step0_results.keys())
    step0_scores = [step0_results[k] for k in sorted_k_values]
    step700_scores = [step700_results[k] for k in sorted_k_values]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_k_values, step0_scores, 'b-o', label='Step 0')
    plt.plot(sorted_k_values, step700_scores, 'r-o', label='Step 700')
    
    # Add labels and title
    plt.xlabel('k')
    plt.ylabel('Pass@k (%)')
    plt.title('Comparison of Pass@k between Step 0 and Step 700')
    plt.legend()
    plt.grid(True)
    
    # Use logarithmic scale for x-axis as k values grow exponentially
    plt.xscale('log')
    plt.xticks(sorted_k_values, [str(k) for k in sorted_k_values], rotation=45)
    
    # Add value labels on the points
    for i, k in enumerate(sorted_k_values):
        plt.annotate(f"{step0_scores[i]}", 
                    (k, step0_scores[i]), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center',
                    fontsize=8)
        plt.annotate(f"{step700_scores[i]}", 
                    (k, step700_scores[i]), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center',
                    fontsize=8)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('pass_at_k_comparison.png')
    plt.show()
    
    # Print the values
    print("Step 0 Pass@k:", step0_results)
    print("Step 700 Pass@k:", step700_results)

if __name__ == "__main__":
    main()