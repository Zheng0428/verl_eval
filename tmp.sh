set -x 

path=/mnt/hdfs/tiktok_aiic/user/tianshun/verl_rl_checkpoints/vllm-v1-0519-on-policy-large-off-policy-maximum-rejection-sampling_rmclipTrue_batch512_ppomini64_rolln64_maxres4096_maxvalres8192_deepscaler_train_simplelr_math_35_train_Qwen2.5-7B_hf_converted
specific_steps=280
get_all_checkpoints() {
    local base_path="$1"
    local specific_steps="$2"
    local checkpoints=()
    
    # If specific steps are provided, only collect those checkpoints
    if [ -n "$specific_steps" ]; then
        IFS=',' read -r -a step_array <<< "$specific_steps"
        for step in "${step_array[@]}"; do
            step_dir="$base_path/global_step_$step"
            if [ -d "$step_dir" ]; then
                checkpoints+=("global_step_$step")
            else
                echo "Warning: Requested step $step does not exist at $step_dir"
            fi
        done
    else
        # Otherwise, collect all checkpoints
        for ckpt_dir in "$base_path"/global_step_*; do
            if [ -d "$ckpt_dir" ]; then
                step_tag=$(basename "$ckpt_dir")        
                current_step=$(echo "$step_tag" | sed 's/global_step_//')
                if (( current_step >= 80 )); then
                    checkpoints+=("$step_tag")
                fi
                checkpoints+=("$step_tag")
            fi
        done
    fi
    
    if [ ${#checkpoints[@]} -eq 0 ]; then
        echo ""
    else
        # Sort the checkpoints to ensure consistent ordering across nodes
        printf "%s\n" "${checkpoints[@]}" | sort -V
    fi
}
# Get all checkpoints

readarray -t all_checkpoints < <(get_all_checkpoints "$path" "$specific_steps")
