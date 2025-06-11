# --- Configuration Variables (Adjust if needed) ---
export INIT_MODEL_PATH="/mnt/hdfs/tiktok_aiic/user/codeai/hf_models/Qwen2.5-7B"
export CHECKPOINT_BASE_PATH="/mnt/hdfs/tiktok_aiic/user/liuqian/verl_rl_checkpoints/vllm-v1-0519-on-policy-large-off-policy-maximum-rejection-sampling_rmclipTrue_batch512_ppomini64_rolln64_maxres4096_maxvalres8192_deepscaler_train_simplelr_math_35_train_Qwen2.5-7B"
export TARGET_BASE_DIR="/mnt/hdfs/tiktok_aiic/user/tianshun/verl_rl_checkpoints/vllm-v1-0519-on-policy-large-off-policy-maximum-rejection-sampling_rmclipTrue_batch512_ppomini64_rolln64_maxres4096_maxvalres8192_deepscaler_train_simplelr_math_35_train_Qwen2.5-7B_hf_converted"

# Define the steps you want to convert
START_STEP=0
END_STEP=0
STEP_INTERVAL=20

# --- Function: Copy Tokenizer Files (No changes needed here) ---
copy_tokenizer_files() {
    local ckpt_path=$1
    local init_model_path=$2
    local files_to_copy=(
        "added_tokens.json"
        "config.json"
        "generation_config.json"
        "special_tokens_map.json"
        "tokenizer_config.json"
        "tokenizer.json"
        "vocab.json"
    )
    if [ -f "$init_model_path/merges.txt" ]; then
        files_to_copy+=("merges.txt")
    fi

    # Create target path, ensuring it exists
    if [ ! -d "$ckpt_path" ]; then
        mkdir -p "$ckpt_path"
        echo "Created checkpoint directory: $ckpt_path" >&2
    else
        echo "Checkpoint directory already exists: $ckpt_path" >&2
    fi

    # Copy each file
    for filename in "${files_to_copy[@]}"; do
        src="$init_model_path/$filename"
        dst="$ckpt_path/$filename"
        if [ -e "$src" ]; then
            cp "$src" "$dst"
            echo "Copied $src to $dst"
        else
            echo "Warning: $src does not exist."
        fi
    done
}

# --- Main Conversion Loop ---
echo "Starting FSDP checkpoint conversion for steps from $START_STEP to $END_STEP with interval $STEP_INTERVAL."

for (( STEP_NUM=START_STEP; STEP_NUM<=END_STEP; STEP_NUM+=STEP_INTERVAL )); do
    CURRENT_STEP_DIR="global_step_${STEP_NUM}"
    CURRENT_CHECKPOINT_PATH="$CHECKPOINT_BASE_PATH/$CURRENT_STEP_DIR/actor" # Adjust if your actual checkpoint files are directly in global_step_XXX
    CURRENT_TARGET_DIR="$TARGET_BASE_DIR/$CURRENT_STEP_DIR" # Save converted model to a step-specific sub-directory

    echo "--- Processing ${CURRENT_STEP_DIR} ---"

    # Run the model conversion script
    python scripts/model_merger.py \
        --backend fsdp \
        --hf_model_path "$INIT_MODEL_PATH" \
        --local_dir "$CURRENT_CHECKPOINT_PATH" \
        --target_dir "$CURRENT_TARGET_DIR"

    # Call copy tokenizer to copy necessary files to the converted model directory
    copy_tokenizer_files "$CURRENT_TARGET_DIR" "$INIT_MODEL_PATH"

    echo "Model conversion and tokenizer copy done for ${CURRENT_STEP_DIR}"
    echo "Converted model saved to: ${CURRENT_TARGET_DIR}"
done

echo "All specified model conversions complete!"