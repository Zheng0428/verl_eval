export CUDA_VISIBLE_DEVICES=0,1

python genrm_train/genrm_rollout.py \
    --model /mnt/hdfs/byte_data_seed_azureb_tteng/user/huangyuzhen/SimpleRL-verl/base_models/DeepSeek-R1-Distill-Qwen-1.5B \
    --input /mnt/bn/tiktok-mm-5/aiic/users/huangyuzhen/verl/genrm_train/data/deepscaler_iter1.1/filtered_data.json \
    --output /mnt/bn/tiktok-mm-5/aiic/users/huangyuzhen/verl/genrm_train/data/deepscaler_iter1.1/generated_data.json \
    --temperature 1.0 --top_p 0.95 --batch_size 1024 --max_tokens 4096 --num_rollouts 24



# python genrm_train/genrm_rollout_w_q.py \
#     --model /mnt/hdfs/byte_data_seed_azureb_tteng/user/huangyuzhen/SimpleRL-verl/checkpoints/r1-1.5b-trn_verifier-lr2e-5-0417 \
#     --input /mnt/bn/tiktok-mm-5/aiic/users/huangyuzhen/verl/genrm_train/data/skywork/filtered_data.json \
#     --output /mnt/bn/tiktok-mm-5/aiic/users/huangyuzhen/verl/genrm_train/data/skywork/generated_data_gen_with_q.json \
#     --temperature 1.0 --top_p 0.95 --batch_size 1024 --max_tokens 8192 --num_rollouts 24