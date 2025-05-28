

TASK="genrm_skywork"
Test_file=/Users/bytedance/Desktop/verl/genrm_train/preli_exp/genrm_eval/xVerify-0.5B-I/${TASK}/val-r1_qwen_32b-xverifier_false_only_w_pred_xVerify-0.5B-I_-1_seed0_t0.1_s0_e-1.jsonl
Org_file=/Users/bytedance/Desktop/verl/examples/simplelr_math_eval/data/${TASK}/val-r1_qwen_32b-xverifier_false_only_w_pred.jsonl

# Test_file=/Users/bytedance/Desktop/verl/genrm_train/preli_exp/genrm_eval/xVerify-3B-Ia/${TASK}/val-r1_qwen_32b-xverifier_false_only_w_pred_xVerify-3B-Ia_-1_seed0_t0.1_s0_e-1.jsonl
# Org_file=/Users/bytedance/Desktop/verl/examples/simplelr_math_eval/data/${TASK}/val-r1_qwen_32b-xverifier_false_only_w_pred.jsonl




python genrm_train/preli_exp/xverify_stats.py \
    --test_file $Test_file \
    --org_file $Org_file
# val-r1_qwen_32b-wo_q_false_only_deepseek-r1_-1_seed0_t0.6_s0_e-1_metrics.json