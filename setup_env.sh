#!/bin/bash
# demo usage: source setup_env.sh
# then run your script

# Default values
DEFAULT_PROJECT_NAME="Reinforcement-Learning"
DEFAULT_WANDB_API_KEY="cb6bcb2df698f249880cb013bcbc07f75f13a457"
DEFAULT_WANDB_OFFICIAL=1
DEFAULT_VLLM_ATTENTION_BACKEND="XFORMERS"
DEFAULT_HDFS_DATA_PATH="/mnt/bn/tiktok-mm-5/aiic/users/tianyu/dataset/RL-dataset"
DEFAULT_HDFS_MODEL_PATH="/mnt/hdfs/tiktok_aiic/user/codeai/hf_models"
# DEFAULT_HDFS_CHECKPOINT_PATH="/mnt/hdfs/tiktok_aiic/user/tianyu/verl_rl_checkpoints"
DEFAULT_HDFS_CHECKPOINT_PATH="/mnt/hdfs/tiktok_aiic/user/tianshun/verl_rl_checkpoints"
DEFAULT_HDFS_LOG_PATH="/mnt/hdfs/tiktok_aiic/user/tianyu/verl_rl_logs"
DEFAULT_RUN_NAME="verl-srpo-v1"

# Use external variables if set, otherwise use defaults
export PROJECT_NAME=${PROJECT_NAME:-$DEFAULT_PROJECT_NAME}
export WANDB_API_KEY=${WANDB_API_KEY:-$DEFAULT_WANDB_API_KEY}
export WANDB_OFFICIAL=${WANDB_OFFICIAL:-$DEFAULT_WANDB_OFFICIAL}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-$DEFAULT_VLLM_ATTENTION_BACKEND}
export HDFS_DATA_PATH=${HDFS_DATA_PATH:-$DEFAULT_HDFS_DATA_PATH}
export HDFS_MODEL_PATH=${HDFS_MODEL_PATH:-$DEFAULT_HDFS_MODEL_PATH}
export HDFS_CHECKPOINT_PATH=${HDFS_CHECKPOINT_PATH:-$DEFAULT_HDFS_CHECKPOINT_PATH}
export HDFS_LOG_PATH=${HDFS_LOG_PATH:-$DEFAULT_HDFS_LOG_PATH}
export RUN_NAME=${RUN_NAME:-$DEFAULT_RUN_NAME}

# Optional: Print current configuration
echo "Current configuration:"
echo "PROJECT_NAME: $PROJECT_NAME"
echo "WANDB_API_KEY: $WANDB_API_KEY"
echo "WANDB_OFFICIAL: $WANDB_OFFICIAL"
echo "VLLM_ATTENTION_BACKEND: $VLLM_ATTENTION_BACKEND"
echo "HDFS_DATA_PATH: $HDFS_DATA_PATH"
echo "HDFS_MODEL_PATH: $HDFS_MODEL_PATH"
echo "HDFS_CHECKPOINT_PATH: $HDFS_CHECKPOINT_PATH"
echo "HDFS_LOG_PATH: $HDFS_LOG_PATH"