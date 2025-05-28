

# ori_dir=$(pwd)
# cd /mnt/bn/tiktok-mm-5/aiic/users/huangyuzhen/ppo-long-cot/train


# export WANDB_API_KEY=78c280e2fa597b45660678c48f3dfe054930af18
# export WANDB_OFFICIAL=1


# bash train_genrm.sh \
#     --train_batch_size 128 \
#     --max_len 8192 \
#     --max_epochs 3 \
#     --dataset_name skywork_deepscaler_merged_train_data \
#     --learning_rate  2e-4 \
#     --model_name DeepSeek-R1-Distill-Qwen-1.5B


# cd $ori_dir


# HDFS_PATH=/mnt/hdfs/byte_data_seed_azureb_tteng/user/huangyuzhen/SimpleRL-verl/base_models

# MODLE_PATH=$HDFS_PATH/DeepSeek-R1-Distill-Qwen-1.5B







bash eval_math_single_model.sh \
     $MODLE_PATH \
     deepseek-r1 \
     $MODLE_PATH/eval_results_temp_0.6 \
     0.6 \
     8192 \
     0.95 \
     "genrm_deepscaler_wo_question,genrm_skywork_wo_question" \
     false