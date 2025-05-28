

bash eval_math_nodes.sh \
    --run_name verl-srpo_v2_new_divide_0.72_maxres16384_maxvalres32000_rollout16_advsrpo_sharpenFalse_srpo_block2_deepscaler_train_simplelr_math_35_train_Qwen2.5-7B \
    --init_model Qwen2.5-7B \
    --template qwen-boxed  \
    --tp_size 2 \
    --add_step_0 true  \
    --temperature 1.0 \
    --top_p 0.7 \
    --max_tokens 32000 \
    --output_dir /opt/tiger/Github-Repo/log_tmp \
    --benchmarks aime24  \
    --n_sampling 32 \
    --convert_model true

bash eval_math_nodes.sh \
    --run_name verl-srpo_v2_new_divide_0.72_maxres16384_maxvalres32000_rollout16_advsrpo_sharpenFalse_srpo_block2_deepscaler_train_simplelr_math_35_train_Qwen2.5-7B \
    --init_model Qwen2.5-7B \
    --template qwen-boxed  \
    --tp_size 2 \
    --add_step_0 true  \
    --temperature 1.0 \
    --top_p 0.95 \
    --max_tokens 32000 \
    --output_dir /opt/tiger/Github-Repo/log_tmp \
    --benchmarks aime24  \
    --n_sampling 32 \
    --convert_model true

bash eval_math_nodes.sh \
    --run_name verl-srpo_v2_new_divide_maxres7500_maxvalres14000_rollout16_advsrpo_sharpenFalse_srpo_block2_deepscaler_train_simplelr_math_35_train_Qwen2.5-32B \
    --init_model Qwen2.5-32B \
    --template qwen-boxed  \
    --tp_size 4 \
    --add_step_0 true  \
    --temperature 1.0 \
    --top_p 0.7 \
    --max_tokens 32000 \
    --output_dir /opt/tiger/Github-Repo/log_tmp \
    --benchmarks aime24  \
    --n_sampling 32 \
    --convert_model true

bash eval_math_nodes.sh \
    --run_name verl-srpo_v2_new_divide_maxres7500_maxvalres14000_rollout16_advsrpo_sharpenFalse_srpo_block2_deepscaler_train_simplelr_math_35_train_Qwen2.5-32B \
    --init_model Qwen2.5-32B \
    --template qwen-boxed  \
    --tp_size 4 \
    --add_step_0 true  \
    --temperature 1.0 \
    --top_p 0.95 \
    --max_tokens 32000 \
    --output_dir /opt/tiger/Github-Repo/log_tmp \
    --benchmarks aime24  \
    --n_sampling 32 \
    --convert_model true