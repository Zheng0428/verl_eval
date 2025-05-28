#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
# Default values
START_STEP=160
END_STEP=160
INTERVAL=10
HF_MODEL_PATH="/mnt/hdfs/byte_data_seed_azureb_tteng/user/huangyuzhen/SimpleRL-verl/base_models/Qwen-2.5-7B"
BASE_PATH="/mnt/hdfs/byte_data_seed_azureb_tteng/user/huangyuzhen/SimpleRL-verl/checkpoints/verl_train_Qwen-2.5-7B_max_response8192_batch1024_ppomini256_rollout8_klloss0.0_entcoef0.0_clipratiohigh0.28_genrm_enableTrue_simplelr_skywork_new"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --start)
      START_STEP="$2"
      shift 2
      ;;
    --end)
      END_STEP="$2"
      shift 2
      ;;
    --interval)
      INTERVAL="$2"
      shift 2
      ;;
    --hf_model_path)
      HF_MODEL_PATH="$2"
      shift 2
      ;;
    --base_path)
      BASE_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Converting models from step $START_STEP to $END_STEP with interval $INTERVAL"

# Loop through steps
for ((step = START_STEP; step <= END_STEP; step += INTERVAL)); do
  echo "Processing step $step..."
  
  # Create paths
  ACTOR_PATH="$BASE_PATH/global_step_$step/actor"
  TARGET_DIR="$ACTOR_PATH/huggingface"
  
  # Run conversion
  python3 scripts/model_merger.py \
    --backend fsdp \
    --hf_model_path "$HF_MODEL_PATH" \
    --local_dir "$ACTOR_PATH" \
    --target_dir "$TARGET_DIR"
  
  echo "Completed conversion for step $step"
done

echo "All conversions completed."
