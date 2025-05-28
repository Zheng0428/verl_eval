#!/bin/bash

# Shell script to run the genrm_stats.py file with command line arguments
# Usage examples are provided below

# Define color codes for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}GenRM Statistics Calculator${NC}"
echo -e "${BLUE}Available options:${NC}"
echo "  --test, -t       Path to the test file with predictions (required)"
echo "  --original, -o   Path to the original file with ground truth data (required)"
echo "  --output, -out   Path to save the results (optional)"
echo "  --rule_based, -r Enable rule-based judging combination (optional)"
echo ""

# Example 1: Basic usage with required arguments
echo -e "${BLUE}Example 1: Basic usage${NC}"
echo "python genrm_train/preli_exp/genrm_stats.py \\"
echo "    --test /path/to/predictions.jsonl \\"
echo "    --original /path/to/ground_truth.jsonl"
echo ""

# Example 2: With output file
echo -e "${BLUE}Example 2: Save results to a file${NC}"
echo "python genrm_train/preli_exp/genrm_stats.py \\"
echo "    --test /path/to/predictions.jsonl \\"
echo "    --original /path/to/ground_truth.jsonl \\"
echo "    --output results/metrics_output.json"
echo ""

# Example 3: With rule-based combination
echo -e "${BLUE}Example 3: Enable rule-based combination${NC}"
echo "python genrm_train/preli_exp/genrm_stats.py \\"
echo "    --test /path/to/predictions.jsonl \\"
echo "    --original /path/to/ground_truth.jsonl \\"
echo "    --rule_based"
echo ""

# Example with actual paths from original code
echo -e "${BLUE}Example with original paths:${NC}"
echo "python genrm_train/preli_exp/genrm_stats.py \\"
echo "    --test \"/Users/bytedance/Desktop/hdfs/r1-distilled-qwen-1.5b/genrm_deepscaler/val-r1_qwen_32b-wo_q_deepseek-r1_-1_seed0_t0.6_s0_e-1.jsonl\" \\"
echo "    --original \"/Users/bytedance/Desktop/verl/examples/simplelr_math_eval/data/genrm_deepscaler/val-r1_qwen_32b-wo_q.jsonl\" \\"
echo "    --rule_based"
echo ""

# Uncomment and modify this section to actually run the script with your desired arguments
# echo -e "${GREEN}Running script with example parameters...${NC}"
# python genrm_train/preli_exp/genrm_stats.py \
#     --test "/Users/bytedance/Desktop/hdfs/r1-distilled-qwen-1.5b/genrm_deepscaler/val-r1_qwen_32b-wo_q_deepseek-r1_-1_seed0_t0.6_s0_e-1.jsonl" \
#     --original "/Users/bytedance/Desktop/verl/examples/simplelr_math_eval/data/genrm_deepscaler/val-r1_qwen_32b-wo_q.jsonl" \
#     --rule_based 