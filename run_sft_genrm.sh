HDFS_CHECKPOINT_PATH=/mnt/hdfs/byte_data_seed_azureb_tteng/user/huangyuzhen/SimpleRL-verl/checkpoints

RUN_NAME="r1-1.5b-genrm-sft"




bash eval_math_single_model.sh \
     $HDFS_CHECKPOINT_PATH/$RUN_NAME \
     deepseek-r1 \
     $HDFS_CHECKPOINT_PATH/$RUN_NAME/eval_results_temp_0.6 \
     0.6 \
     8192 \
     0.95 \
     "genrm_deepscaler_wo_question,genrm_skywork_wo_question" \
     false

python genrm_train/stats_genrm.py \
    $HDFS_CHECKPOINT_PATH/$RUN_NAME/eval_results_temp_0.6 