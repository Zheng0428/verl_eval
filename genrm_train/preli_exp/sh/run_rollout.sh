MODEL="meta-llama/Llama-2-7b-chat-hf"
DATASETS="
genrm_train/preli_exp/process_for_rollout/DeepScaleR-Preview-Dataset.jsonl 
genrm_train/preli_exp/raw_data/math_train.jsonl
"

bash genrm_train/preli_exp/run_rollout.sh $MODEL "$DATASETS"