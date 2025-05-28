# pip install deepspeed==0.15.0
# pip install --no-deps -e . 


export WANDB_API_KEY=78c280e2fa597b45660678c48f3dfe054930af18
export WANDB_OFFICIAL=1



RUN_NAME="r1-1.5b-genrm-sft"
HDFS_HOME="/mnt/hdfs/byte_data_seed_azureb_tteng/user/huangyuzhen/SimpleRL-verl"
HDFS_MODEL_PATH=/mnt/hdfs/byte_data_seed_azureb_tteng/user/huangyuzhen/SimpleRL-verl/base_models
BASE_MODEL="DeepSeek-R1-Distill-Qwen-1.5B"
HDFS_CHECKPOINT_PATH=/mnt/hdfs/byte_data_seed_azureb_tteng/user/huangyuzhen/SimpleRL-verl/checkpoints
DATASET_BASE_PATH=/mnt/bn/tiktok-mm-5/aiic/users/huangyuzhen/verl/genrm_train/data
DATASET_NAME="skywork_deepscaler_merged_train_data"
TRAIN_BATCH_SIZE=128
LEARNING_RATE=2e-4
MAX_EPOCHS=3

# Function to generate suffix based on parameters
generate_suffix() {
  local suffix=""
  local dataset_provided=false
  local suffix_provided=false

  while [[ "$#" -gt 0 ]]; do
    case $1 in
      --train_batch_size) suffix+="_batch$2"; shift 2 ;;
      --learning_rate) suffix+="_lr$2"; shift 2 ;;
      --max_epochs) suffix+="_epochs$2"; shift 2 ;;
      --dataset_name) suffix+="_$2"; dataset_provided=true; shift 2 ;;
      --model_name) suffix+="_$2"; model_provided=true; shift 2 ;;
      --max_len) suffix+="_maxlen$2"; shift 2 ;;
      --suffix) input_suffix="$2"; suffix_provided=true; shift 2 ;;
      *) shift ;;
    esac
  done

  if [ "$dataset_provided" = false ]; then
    suffix+="_$DATASET_NAME"
  fi

  if [ "$model_provided" = false ]; then
    suffix+="_$BASE_MODEL"
  fi

  if [ "$suffix_provided" = true ]; then
    suffix+="_$input_suffix"
  fi
  
  echo "$suffix"
}

while [[ "$#" -gt 0 ]]; do
  echo "Processing: $1"
  case "$1" in
    --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
    --max_epochs) MAX_EPOCHS="$2"; shift 2 ;;
    --dataset_name) DATASET_NAME="$2"; shift 2 ;;
    --model_name) BASE_MODEL="$2"; shift 2 ;;
    --max_len) MAX_LEN="$2"; shift 2 ;;
    --suffix) SUFFIX="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Generate suffix and create run name if not specified
if [ -z "$RUN_NAME" ]; then
  GENERATED_SUFFIX=$(generate_suffix "$@")
  RUN_NAME=$RUN_NAME$GENERATED_SUFFIX
fi

echo "RUN_NAME: $RUN_NAME"
echo "Training with parameters:"
echo "- Model: $BASE_MODEL"
echo "- Dataset: $DATASET_NAME"
echo "- Batch Size: $TRAIN_BATCH_SIZE"
echo "- Learning Rate: $LEARNING_RATE"
echo "- Max Epochs: $MAX_EPOCHS"

accelerate launch \
   --config_file recipes/deepspeed_zero3.yaml openrlhf/cli/train_sft.py \
   --max_len $MAX_LEN \
   --dataset $DATASET_BASE_PATH/$DATASET_NAME.json \
   --input_key input \
   --output_key target \
   --train_batch_size $TRAIN_BATCH_SIZE \
   --micro_train_batch_size 1 \
   --max_samples 500000 \
   --pretrain $HDFS_MODEL_PATH/$BASE_MODEL \
   --save_path $HDFS_CHECKPOINT_PATH/$RUN_NAME \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs $MAX_EPOCHS \
   --bf16 \
   --flash_attn \
   --learning_rate $LEARNING_RATE \
   --gradient_checkpointing \
   --use_wandb $WANDB_API_KEY \
   --wandb_run_name $RUN_NAME \
   --ckpt_path $HDFS_CHECKPOINT_PATH/$RUN_NAME \
   --max_ckpt_num 10