#!/bin/bash

# HDFS_PATH=/mnt/hdfs/byte_data_seed_azureb_tteng/user/huangyuzhen/SimpleRL-verl/base_models
HDFS_PATH=/mnt/bn/tiktok-mm-5/aiic/users/huangyuzhen
RUN_NAME=DeepSeek-R1-Distill-Qwen-32B
MODEL_PATH=$HDFS_PATH/$RUN_NAME

EVAL_FOLDER=eval_results_rollout
LOG_DIR="/mnt/bn/tiktok-mm-5/aiic/users/huangyuzhen/verl/rollout_log"

# Create the main log directory if it doesn't exist
mkdir -p ${LOG_DIR}

# Define the datasets and available GPUs
# DATASETS=("rollout_math" "rollout_orz" "rollout_skywork" "rollout_deepscaler")
DATASETS=("rollout_skywork" "rollout_deepscaler")
GPUS=(0 1 2 3 4 5 6 7)  # Adjust based on your available GPUs
GPUS_PER_DATASET=4      # Allocate 2 GPUs per dataset

# Make sure we have enough GPUs for datasets
if [ ${#GPUS[@]} -lt $((${#DATASETS[@]} * GPUS_PER_DATASET)) ]; then
    echo "Warning: Not enough GPUs to give $GPUS_PER_DATASET GPUs to each dataset. Some datasets might not run."
fi

# Launch each dataset evaluation with multiple GPUs
for i in "${!DATASETS[@]}"; do
    DATASET=${DATASETS[$i]}
    
    # Create a dataset-specific log directory
    DATASET_LOG_DIR="${LOG_DIR}/${DATASET}"
    mkdir -p ${DATASET_LOG_DIR}
    
    # Generate a timestamp for the log file
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="${DATASET_LOG_DIR}/eval_${TIMESTAMP}.log"
    
    # Calculate the starting index for this dataset's GPUs
    START_IDX=$((i * GPUS_PER_DATASET))
    
    # Check if we have enough GPUs for this dataset
    if [ $START_IDX -ge ${#GPUS[@]} ]; then
        echo "Not enough GPUs available for dataset $DATASET. Skipping."
        continue
    fi
    
    # Determine how many GPUs we can actually assign to this dataset
    AVAIL_GPUS=0
    GPU_LIST=""
    
    for j in $(seq 0 $((GPUS_PER_DATASET-1))); do
        GPU_IDX=$((START_IDX + j))
        if [ $GPU_IDX -lt ${#GPUS[@]} ]; then
            if [ -z "$GPU_LIST" ]; then
                GPU_LIST="${GPUS[$GPU_IDX]}"
            else
                GPU_LIST="$GPU_LIST,${GPUS[$GPU_IDX]}"
            fi
            AVAIL_GPUS=$((AVAIL_GPUS + 1))
        fi
    done
    
    echo "Starting evaluation for $DATASET using $AVAIL_GPUS GPUs: $GPU_LIST"
    echo "Logging output to: $LOG_FILE"
    
    # Export the specific GPUs for this process and redirect output to log file
    (
        echo "=== Evaluation for $DATASET started at $(date) ===" 
        echo "Using GPUs: $GPU_LIST"
        
        CUDA_VISIBLE_DEVICES=$GPU_LIST bash eval_math_single_model.sh \
            $MODEL_PATH \
            deepseek-r1 \
            $MODEL_PATH/$EVAL_FOLDER \
            1.0 \
            16000 \
            0.95 \
            "$DATASET" \
            false \
            8 \
            size_1000_train
            
        echo "=== Evaluation for $DATASET completed at $(date) ==="
    ) &> ${LOG_FILE} &
    
    # Create a symlink to the latest log file for easier access
    ln -sf ${LOG_FILE} ${DATASET_LOG_DIR}/latest.log
    
    # Store process ID for later reference
    PIDS[$i]=$!
    
    # Small delay to prevent race conditions
    sleep 2
done

# Wait for all evaluations to complete
echo "All evaluations started. Waiting for completion..."
for PID in ${PIDS[*]}; do
    if [[ ! -z "$PID" ]]; then
        wait $PID
        echo "Process $PID completed."
    fi
done

echo "All evaluations completed!"
echo "Log files are available in: ${LOG_DIR}"

# Uncomment to run statistics after all evaluations
# pip install scikit-learn
# python genrm_train/stats_genrm.py \
#     $MODEL_PATH/$EVAL_FOLDER 