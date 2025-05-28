source setup_env.sh

export GPUS_PER_NODE=8
export NNODES=$ARNOLD_WORKER_NUM
export NODE_RANK=$ARNOLD_ID
export MASTER_ADDR=$ARNOLD_WORKER_0_HOST
export MASTER_PORT=$ARNOLD_WORKER_0_PORT

torchrun --nproc-per-node $GPUS_PER_NODE \
    --master-addr $MASTER_ADDR \
    --node-rank $NODE_RANK \
    --master_port $MASTER_PORT \
    --nnodes $NNODES -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HDFS_DATA_PATH/openhermes/train.parquet \
    data.val_files=$HDFS_DATA_PATH/openhermes/test.parquet \
    data.train_batch_size=512 \
    data.micro_batch_size_per_gpu=4 \
    data.prompt_key=question \
    data.response_key=answer \
    data.max_length=2048 \
    data.truncation=right \
    model.partial_pretrain=$HDFS_MODEL_PATH/Mistral-Small-24B-Base-2501 \
    trainer.project_name=openhermes-sft \
    trainer.default_local_dir=$HDFS_CHECKPOINT_PATH/Openhermes-Mistral-24B-Qwen-Chat \
    trainer.experiment_name=Openhermes-Mistral-24B-Qwen-Chat \
    trainer.total_epochs=4 \
    trainer.save_freq=10 \
    trainer.logger=['console','wandb']
