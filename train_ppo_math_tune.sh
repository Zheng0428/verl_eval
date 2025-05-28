# Default values
TRAIN_BATCH_SIZE=256
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=2048
LEARNING_RATE=5e-7
CRITIC_LEARNING_RATE=9e-6
CRITIC_WARMUP=0
PPO_MINI_BATCH_SIZE=256
# per GPU
PPO_MICRO_BATCH_SIZE=4
CLIP_RATIO=0.2
KL_LOSS_COEF=0.001
ENTROPY_COEFFIENT=0.001
KL_LOSS_TYPE="low_var_kl"
TEMPERATURE=1.0
MIN_P=0.0
TOP_P=1.0
TOP_K=-1
# per GPU
LOG_PROB_MICRO_BATCH_SIZE=20
ROLLOUT_N=8
KL_COEF=0.001
TOTAL_EPOCHS=20
DATASET_NAME=simplelr_math_35
ROLLOUT_GPU_MEMORY_UTIL=0.6
ACTOR_OPTIMIZER_OFFLOAD=False
ACTOR_PARAMETER_OFFLOAD=False
MODEL_NAME=Qwen2.5-7B
SAVE_FREQ=20
TEST_FREQ=20
REMOVE_CLIP=False
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=2
MICRO_ROLLOUT_BATCH_SIZE=1024
REMOVE_PREVIOUS_CKPT=False

generate_suffix() {
  local suffix=""
  local dataset_provided=false
  local model_provided=false
  local suffix_provided=false

  while [[ "$#" -gt 0 ]]; do
    case $1 in
      --train_batch_size) suffix+="_batch$2"; shift 2 ;;
      --max_prompt_length) suffix+="_max_prompt$2"; shift 2 ;;
      --max_response_length) suffix+="_max_response$2"; shift 2 ;;
      --learning_rate) suffix+="_lr$2"; shift 2 ;;
      --critic_learning_rate) suffix+="_criticlr$2"; shift 2 ;;
      --critic_warmup) suffix+="_criticwarmup$2"; shift 2 ;;
      --ppo_mini_batch_size) suffix+="_ppomini$2"; shift 2 ;;
      --kl_loss_coef) suffix+="_klcoef$2"; shift 2 ;;
      --entropy_coeffient) suffix+="_entcoef$2"; shift 2 ;;
      --clip_ratio) suffix+="_clipratio$2"; shift 2 ;;
      --kl_loss_type) suffix+="_kltype$2"; shift 2 ;;
      --temperature) suffix+="_temp$2"; shift 2 ;;
      --top_p) suffix+="_topp$2"; shift 2 ;;
      --top_k) suffix+="_topk$2"; shift 2 ;;
      --min_p) suffix+="_minp$2"; shift 2 ;;
      --rollout_n) suffix+="_rollout$2"; shift 2 ;;
      --kl_coef) suffix+="_klcontrol$2"; shift 2 ;;
      --total_epochs) suffix+="_epochs$2"; shift 2 ;;
      --dataset_name) suffix+="_$2"; dataset_provided=true; shift 2 ;;
      --model_name) suffix+="_$2"; model_provided=true; shift 2 ;;
      --reword_function_type) suffix+="_reward_type$2"; shift 2 ;;
      --format_penalty_value) suffix+="_format_penalty$2"; shift 2 ;;
      --remove_clip) suffix+="_remove_clip$2"; shift 2 ;;
      --suffix) input_suffix="$2"; suffix_provided=true; shift 2 ;;
      *) shift ;;
    esac
  done

  if [ "$dataset_provided" = false ]; then
    suffix+="_$DATASET_NAME"
  fi

  if [ "$model_provided" = false ]; then
    suffix+="_$MODEL_NAME"
  fi

  if [ "$suffix_provided" = true ]; then
    suffix+="_$input_suffix"
  fi
  
  echo "$suffix"
}

echo "Arguments received: $@"

# Generate a unique suffix based on the input arguments
SUFFIX=$(generate_suffix "$@")
RUN_NAME="$RUN_NAME$SUFFIX"
LOG_FILE_PATH="$HDFS_LOG_PATH/$RUN_NAME.log"

echo "RUN_NAME: $RUN_NAME"
echo "LOG_FILE_PATH: $LOG_FILE_PATH"

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
  echo "Processing: $1"
  case "$1" in
    --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --max_prompt_length) MAX_PROMPT_LENGTH="$2"; shift 2 ;;
    --max_response_length) MAX_RESPONSE_LENGTH="$2"; shift 2 ;;
    --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
    --critic_learning_rate) CRITIC_LEARNING_RATE="$2"; shift 2 ;;
    --critic_warmup) CRITIC_WARMUP="$2"; shift 2 ;;
    --ppo_mini_batch_size) PPO_MINI_BATCH_SIZE="$2"; shift 2 ;;
    --ppo_micro_batch_size) PPO_MICRO_BATCH_SIZE="$2"; shift 2 ;;
    --kl_loss_coef) KL_LOSS_COEF="$2"; shift 2 ;;
    --entropy_coeffient) ENTROPY_COEFFIENT="$2"; shift 2 ;;
    --clip_ratio) CLIP_RATIO="$2"; shift 2 ;;
    --kl_loss_type) KL_LOSS_TYPE="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --top_p) TOP_P="$2"; shift 2 ;;
    --top_k) TOP_K="$2"; shift 2 ;;
    --min_p) MIN_P="$2"; shift 2 ;;
    --log_prob_micro_batch_size) LOG_PROB_MICRO_BATCH_SIZE="$2"; shift 2 ;;
    --rollout_n) ROLLOUT_N="$2"; shift 2 ;;
    --rollout_gpu_memory_util) ROLLOUT_GPU_MEMORY_UTIL="$2"; shift 2 ;;
    --rollout_tp) ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="$2"; shift 2 ;;
    --micro_rollout_batch_size) MICRO_ROLLOUT_BATCH_SIZE="$2"; shift 2 ;;
    --kl_coef) KL_COEF="$2"; shift 2 ;;
    --actor_optimizer_offload) ACTOR_OPTIMIZER_OFFLOAD="$2"; shift 2 ;;
    --actor_parameter_offload) ACTOR_PARAMETER_OFFLOAD="$2"; shift 2 ;;
    --total_epochs) TOTAL_EPOCHS="$2"; shift 2 ;;
    --dataset_name) DATASET_NAME="$2"; shift 2 ;;
    --model_name) MODEL_NAME="$2"; shift 2 ;;
    --save_freq) SAVE_FREQ="$2"; shift 2 ;;
    --test_freq) TEST_FREQ="$2"; shift 2 ;;
    --remove_clip) REMOVE_CLIP="$2"; shift 2 ;;
    --remove_previous_ckpt) REMOVE_PREVIOUS_CKPT="$2"; shift 2 ;;
    --suffix) SUFFIX="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "RUN_NAME: $RUN_NAME"
echo "LOG_FILE_PATH: $LOG_FILE_PATH"
echo "Training with the following parameters:" | tee -a $LOG_FILE_PATH
echo "Train Batch Size: $TRAIN_BATCH_SIZE" | tee -a $LOG_FILE_PATH
echo "Val Batch Size: $VAL_BATCH_SIZE" | tee -a $LOG_FILE_PATH
echo "Max Prompt Length: $MAX_PROMPT_LENGTH" | tee -a $LOG_FILE_PATH
echo "Max Response Length: $MAX_RESPONSE_LENGTH" | tee -a $LOG_FILE_PATH
echo "Learning Rate: $LEARNING_RATE" | tee -a $LOG_FILE_PATH
echo "PPO Mini Batch Size: $PPO_MINI_BATCH_SIZE" | tee -a $LOG_FILE_PATH
echo "PPO Micro Batch Size: $PPO_MICRO_BATCH_SIZE" | tee -a $LOG_FILE_PATH 
echo "Micro Rollout Batch Size: $MICRO_ROLLOUT_BATCH_SIZE" | tee -a $LOG_FILE_PATH
echo "Critic Learning Rate: $CRITIC_LEARNING_RATE" | tee -a $LOG_FILE_PATH
echo "Critic Warmup: $CRITIC_WARMUP" | tee -a $LOG_FILE_PATH
echo "KL Loss Coefficient: $KL_LOSS_COEF" | tee -a $LOG_FILE_PATH
echo "KL Loss Type: $KL_LOSS_TYPE" | tee -a $LOG_FILE_PATH
echo "Temperature: $TEMPERATURE" | tee -a $LOG_FILE_PATH
echo "Rollout N: $ROLLOUT_N" | tee -a $LOG_FILE_PATH
echo "KL Coefficient: $KL_COEF" | tee -a $LOG_FILE_PATH
echo "Total Epochs: $TOTAL_EPOCHS" | tee -a $LOG_FILE_PATH
echo "Dataset Name: $DATASET_NAME" | tee -a $LOG_FILE_PATH
echo "Model Name: $MODEL_NAME" | tee -a $LOG_FILE_PATH
echo "Remove Clip: $REMOVE_CLIP" | tee -a $LOG_FILE_PATH


max_num_batched_tokens=$(expr $MAX_PROMPT_LENGTH + $MAX_RESPONSE_LENGTH + 1000)
# Example of using the variables
sleep 3
source setup_env.sh

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
    data.train_files=$HDFS_DATA_PATH/$DATASET_NAME/train.parquet \
    data.val_files=$HDFS_DATA_PATH/$DATASET_NAME/test.parquet \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.model.path=$HDFS_MODEL_PATH/$MODEL_NAME \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFFIENT \
    actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAMETER_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    actor_rollout_ref.rollout.top_k=$TOP_K \
    actor_rollout_ref.rollout.min_p=$MIN_P \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTIL \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.rollout.micro_rollout_batch_size=$MICRO_ROLLOUT_BATCH_SIZE \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    critic.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
    critic.optim.lr=$CRITIC_LEARNING_RATE \
    critic.model.path=$HDFS_MODEL_PATH/$MODEL_NAME \
    trainer.critic_warmup=$CRITIC_WARMUP \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$ARNOLD_WORKER_NUM \
    trainer.remove_clip=$REMOVE_CLIP \
    trainer.val_generations_to_log_to_wandb=64 \
    trainer.remove_previous_ckpt_in_save=$REMOVE_PREVIOUS_CKPT \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.default_local_dir=$HDFS_CHECKPOINT_PATH/$RUN_NAME \
    trainer.total_epochs=$TOTAL_EPOCHS