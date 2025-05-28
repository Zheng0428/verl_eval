"""
Configuration file for GenRM statistics.
Contains mappings for model names to test files and task names to original files.
"""

import os

# Base directories - update these to match your environment
VERL_BASE = "/Users/bytedance/Desktop/verl"
EVAL_BASE = "/Users/bytedance/Desktop/verl/genrm_train/preli_exp/genrm_eval"
# # suffix = "false_only" # "whole_set"
# suffix = "whole_set"
# Model name to test file mapping with task-specific paths
MODEL_TEST_FILES = {
    # DeepSeek models
    "DeepSeek-R1-Distill-Qwen-1.5B": "val-r1_qwen_32b-wo_q_{suffix}_deepseek-r1_-1_seed0_t0.6_s0_e-1.jsonl",
    "DeepSeek-R1-Distill-Qwen-7B": "val-r1_qwen_32b-wo_q_{suffix}_deepseek-r1_-1_seed0_t0.6_s0_e-1.jsonl",
    
    # xVerify models - using the w_pred versions
    "xVerify-0.5B-I": "val-r1_qwen_32b-xverifier_{suffix}_w_pred_xVerify-0.5B-I_-1_seed0_t0.1_s0_e-1.jsonl",
    "xVerify-3B-Ia": "val-r1_qwen_32b-xverifier_{suffix}_w_pred_xVerify-3B-Ia_-1_seed0_t0.1_s0_e-1.jsonl",
    
    # Qwen models
    "Qwen2.5-1.5B-Instruct": "val-r1_qwen_32b-wo_q_{suffix}_qwen-boxed_-1_seed0_t0.0_p1_s0_e-1.jsonl",
    "Qwen2.5-7B-Instruct": "val-r1_qwen_32b-wo_q_{suffix}_qwen-boxed_-1_seed0_t0.0_p1_s0_e-1.jsonl",
    "Qwen2.5-Math-7B-Instruct": "val-r1_qwen_32b-wo_q_{suffix}_qwen-boxed_-1_seed0_t0.0_p1_s0_e-1.jsonl",
    "Qwen2.5-Math-1.5B-Instruct": "val-r1_qwen_32b-wo_q_{suffix}_qwen-boxed_-1_seed0_t0.0_p1_s0_e-1.jsonl",
    
    # Wenhu verifier models - note there's a difference between skywork (wo_q) and others (w_q)
    "tiger-verifier": "val-r1_qwen_32b-w_q_{suffix}_wenhu_verifier_wenhu-verifier_-1_seed0_t0.0_p1_s0_e-1.jsonl",
    "r1-1.5b-trn_verifier-lr1e-4-0417-3epoch": "val-r1_qwen_32b-w_q_{suffix}_deepseek-r1_-1_seed0_t0.0_p1_s0_e-1.jsonl"
}

# Task name to original file mapping
TASK_ORG_FILES = {
    "genrm_skywork": os.path.join(VERL_BASE, "examples/simplelr_math_eval/data/genrm_skywork/val-r1_qwen_32b-wo_q_{suffix}.jsonl"),
    "genrm_orz": os.path.join(VERL_BASE, "examples/simplelr_math_eval/data/genrm_orz/val-r1_qwen_32b-wo_q_{suffix}.jsonl"),
    "genrm_deepscaler": os.path.join(VERL_BASE, "examples/simplelr_math_eval/data/genrm_deepscaler/val-r1_qwen_32b-wo_q_{suffix}.jsonl"),
    "genrm_math": os.path.join(VERL_BASE, "examples/simplelr_math_eval/data/genrm_math/val-r1_qwen_32b-wo_q_{suffix}.jsonl"),
}

# Original example counts for each task (from your script)
TASK_EXAMPLE_COUNTS = {
    "genrm_skywork": 2000,
    "genrm_orz": 2000,
    "genrm_deepscaler": 2000,
    "genrm_math": 2000,
}

# Special handling for different model types (if needed for processing)
MODEL_TYPES = {
    "deepseek": ["DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek-R1-Distill-Qwen-7B", "r1-1.5b-trn_verifier-lr1e-4-0417-3epoch"],
    "xverify": ["xVerify-0.5B-I", "xVerify-3B-Ia"],
    "qwen": [
        "Qwen2.5-1.5B-Instruct", 
        "Qwen2.5-7B-Instruct", 
        "Qwen2.5-Math-7B-Instruct", 
        "Qwen2.5-Math-1.5B-Instruct"
    ],
    "wenhu": ["tiger-verifier"],
}

# Create mapping from model name to model type
MODEL_TO_TYPE_MAP = {}
for model_type, model_names in MODEL_TYPES.items():
    for model_name in model_names:
        MODEL_TO_TYPE_MAP[model_name] = model_type




