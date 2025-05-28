

# TASK="genrm_skywork"
# Test_file=/Users/bytedance/Desktop/verl/genrm_train/preli_exp/genrm_eval/DeepSeek-R1-Distill-Qwen-1.5B/${TASK}/val-r1_qwen_32b-wo_q_false_only_deepseek-r1_-1_seed0_t0.6_s0_e-1.jsonl
# Org_file=/Users/bytedance/Desktop/verl/examples/simplelr_math_eval/data/${TASK}/val-r1_qwen_32b-wo_q_false_only.jsonl






# python genrm_train/preli_exp/genrm_stats_fnr.py \
#     --test_file $Test_file \
#     --org_file $Org_file \
#     --org_example_num 2000
# val-r1_qwen_32b-wo_q_false_only_deepseek-r1_-1_seed0_t0.6_s0_e-1_metrics.json




# TASK="genrm_orz"
# Test_file=/Users/bytedance/Desktop/verl/genrm_train/preli_exp/genrm_eval/xVerify-0.5B-I/${TASK}/val-r1_qwen_32b-xverifier_false_only_w_pred_xVerify-0.5B-I_-1_seed0_t0.1_s0_e-1.jsonl
# Org_file=/Users/bytedance/Desktop/verl/examples/simplelr_math_eval/data/${TASK}/val-r1_qwen_32b-wo_q_false_only.jsonl


# python genrm_train/preli_exp/xverify_stats.py \
#     --test_file $Test_file \
#     --org_file $Org_file \
#     --org_example_num 2000

# TASK="genrm_deepscaler"
# Test_file=/Users/bytedance/Desktop/verl/genrm_train/preli_exp/genrm_eval/xVerify-3B-Ia/${TASK}/val-r1_qwen_32b-xverifier_false_only_w_pred_xVerify-3B-Ia_-1_seed0_t0.1_s0_e-1.jsonl
# Org_file=/Users/bytedance/Desktop/verl/examples/simplelr_math_eval/data/${TASK}/val-r1_qwen_32b-wo_q_false_only.jsonl



# TASK="genrm_skywork"
# Test_file=/Users/bytedance/Desktop/verl/genrm_train/preli_exp/genrm_eval/Qwen2.5-1.5B-Instruct/${TASK}/val-r1_qwen_32b-wo_q_false_only_qwen-boxed_-1_seed0_t0.6_p0.95_s0_e-1.jsonl
# Org_file=/Users/bytedance/Desktop/verl/examples/simplelr_math_eval/data/${TASK}/val-r1_qwen_32b-wo_q_false_only.jsonl


# TASK="genrm_math"
# Test_file=/Users/bytedance/Desktop/verl/genrm_train/preli_exp/genrm_eval/Qwen2.5-7B-Instruct/${TASK}/val-r1_qwen_32b-wo_q_false_only_qwen-boxed_-1_seed0_t0.6_p0.95_s0_e-1.jsonl
# Org_file=/Users/bytedance/Desktop/verl/examples/simplelr_math_eval/data/${TASK}/val-r1_qwen_32b-wo_q_false_only.jsonl


# TASK="genrm_orz"
# Test_file=/Users/bytedance/Desktop/verl/genrm_train/preli_exp/genrm_eval/Qwen2.5-math-7B-Instruct/${TASK}/val-r1_qwen_32b-wo_q_false_only_qwen25-math-cot_-1_seed0_t0.6_p0.95_s0_e-1.jsonl
# Org_file=/Users/bytedance/Desktop/verl/examples/simplelr_math_eval/data/${TASK}/val-r1_qwen_32b-wo_q_false_only.jsonl

# TASK="genrm_deepscaler"
# Test_file=/Users/bytedance/Desktop/verl/genrm_train/preli_exp/genrm_eval/Qwen2.5-math-1.5B-Instruct/${TASK}/val-r1_qwen_32b-wo_q_false_only_qwen25-math-cot_-1_seed0_t0.6_p0.95_s0_e-1.jsonl
# Org_file=/Users/bytedance/Desktop/verl/examples/simplelr_math_eval/data/${TASK}/val-r1_qwen_32b-wo_q_false_only.jsonl



# python genrm_train/preli_exp/genrm_stats_fnr.py \
#     --test_file $Test_file \
#     --org_file $Org_file \
#     --org_example_num 2000



TASK="genrm_skywork"
Test_file=/Users/bytedance/Desktop/verl/genrm_train/preli_exp/genrm_eval/tiger-verifier/${TASK}/val-r1_qwen_32b-wo_q_false_wenhu_verifier_wenhu-verifier_-1_seed0_t0.0_p1_s0_e-1.jsonl
Org_file=/Users/bytedance/Desktop/verl/examples/simplelr_math_eval/data/${TASK}/val-r1_qwen_32b-w_q_false_only.jsonl

python genrm_train/preli_exp/wenhu_stats.py \
    --test_file $Test_file \
    --org_file $Org_file \
    --org_example_num 2000