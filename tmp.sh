HDFS_PATH=/mnt/hdfs/byte_data_seed_azureb_tteng/user/huangyuzhen/SimpleRL-verl/checkpoints

MODLE_PATH=$HDFS_PATH/r1-1.5b-trn_verifier-lr2e-5-0417

bash eval_math_single_model.sh \
     $MODLE_PATH \
     deepseek-r1 \
     $MODLE_PATH/eval_results_temp_0.6 \
     0.6 \
     8192 \
     0.95 \
     "genrm" \
     false