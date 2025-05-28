

HDFS_PATH=/mnt/hdfs/byte_data_seed_azureb_tteng/user/huangyuzhen/SimpleRL-verl/base_models
RUN_NAME=DeepSeek-R1-Distill-Qwen-7B
MODLE_PATH=$HDFS_PATH/$RUN_NAME

EVAL_FOLDER=eval_results_rollout
export CUDA_VISIBLE_DEVICES=0,1



#"rollout_math,rollout_orz,rollout_skywork,rollout_deepscaler" \
bash eval_math_single_model.sh \
     $MODLE_PATH \
     deepseek-r1 \
     $MODLE_PATH/$EVAL_FOLDER \
     1.0 \
     16000 \
     0.95 \
     "rollout_math" \
     false \
     16 \
     train

# pip install scikit-learn
# python genrm_train/stats_genrm.py \
#     $MODLE_PATH/$EVAL_FOLDER
