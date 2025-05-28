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
    k_values = [1, 2, 3, 4, 5, 8, 10, 16, 20, 32, 40, 50, 64, 100, 128, 200, 256, 300, 400, 512, 600, 700,800, 900,1000, 1024]
    