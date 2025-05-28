

TASK="genrm_skywork"
Test_file=/Users/bytedance/Desktop/verl/genrm_train/preli_exp/genrm_eval/DeepSeek-R1-Distill-Qwen-7B/${TASK}/val-r1_qwen_32b-wo_q_false_only_deepseek-r1_-1_seed0_t0.6_s0_e-1.jsonl
Org_file=/Users/bytedance/Desktop/verl/examples/simplelr_math_eval/data/${TASK}/val-r1_qwen_32b-wo_q_false_only.jsonl




python genrm_train/preli_exp/genrm_stats.py \
    --test_file $Test_file \
    --org_file $Org_file
# val-r1_qwen_32b-wo_q_false_only_deepseek-r1_-1_seed0_t0.6_s0_e-1_metrics.json