#! /bin/bash
#! /bin/bash

USER_ENV=`whoami`
# export RAY_LOG_DIR="/opt/tiger/ray"
# set -x
export NCCL_DEBUG=DEBUG
export RAY_BACKEND_LOG_LEVEL=debug
export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_DEDUP_LOGS=1
RUN_SCRIPT=$1
export WANDB_OFFICIAL=1
export WANDB_API_KEY=78c280e2fa597b45660678c48f3dfe054930af18
### prepare basic env
### we can asssume in ARNOLD_MIXED_RAY mode, only supposed role will execute this script
role=$ARNOLD_ROLE
role_uppercase=${role^^}

### prepare configs for ray start
server_hosts=${ARNOLD_WORKER_HOSTS}
# if [[ $role == "head" ]]; then
#   worker_id=-1
#   IFS=':' SERVER_CONFIGS=($server_hosts)
#   unset IFS
#   num_cpus=${ARNOLD_HEAD_CPU} 
#   num_gpus=${ARNOLD_HEAD_GPU} 
#   num_memory=$(($ARNOLD_HEAD_MEM >> 10))
#   memory=$(($ARNOLD_HEAD_MEM * 1024 ** 2))
#   obj_store_mem_limit=$(($ARNOLD_HEAD_MEM * 450 * 1024))
#   if [[ "$ARNOLD_HEAD_GPU" -eq "0" ]]; then
#     obj_store_mem=$((64 * 1024 ** 3))
#   else
#     obj_store_mem=$((${ARNOLD_HEAD_GPU:-1} * 64 * 1024 ** 3))
#   fi
# else
worker_id=$ARNOLD_ID
eval 'num_cpus="${ARNOLD_'${role_uppercase}'_CPU}"'
eval 'num_gpus="${ARNOLD_'${role_uppercase}'_GPU}"'
eval 'memory_tmp="${ARNOLD_'${role_uppercase}'_MEM}"'
num_memory=$(($memory_tmp >> 10))
memory=$(($memory_tmp * 1024 ** 2))
obj_store_mem_limit=$(($memory_tmp * 450 * 1024))
if [[ -n "$WORKER_OBJECT_STORE_MEM" ]]; then
obj_store_mem=$(($WORKER_OBJECT_STORE_MEM * 1024 ** 3))
else
obj_store_mem=$((8 * 1024 ** 3))
#   fi
fi

if [[ "$obj_store_mem" -gt "$obj_store_mem_limit" ]]; then 
  obj_store_mem=$obj_store_mem_limit
fi
customized_resource='--resources={"'$role'_cpu":'$num_cpus',"'$role'_gpu":'$num_gpus',"'$role'_memory":'$num_memory',"'$role'_'$ARNOLD_ID'_cpu":'$num_cpus',"'$role'_'$ARNOLD_ID'_gpu":'$num_gpus',"'$role'_'$ARNOLD_ID'_memory":'$num_memory'}'

if [[ $server_hosts =~ "," ]]
then
echo 'may be in using vscode'
IFS=',' server_hosts=($server_hosts)
unset IFS
fi

node_addr=$BYTED_HOST_IP

if [[ $server_hosts =~ "]:" ]] ## Checks if the address is IPv6 format
then
SERVER_IP=`echo $server_hosts | awk -F ']:' '{print $1}'`
SERVER_IP+="]"
SERVER_PORT=`echo $server_hosts | awk -F ']:' '{print $2}'`
node_addr="[$MY_HOST_IPV6]" ## Set node address in IPv6 format
else
IFS=':' SERVER_CONFIGS=($server_hosts)
unset IFS
SERVER_IP=${SERVER_CONFIGS[0]}
SERVER_PORT=${SERVER_CONFIGS[1]}
fi

echo $SERVER_IP
echo $SERVER_PORT
echo $node_addr

# in order to avoid mixing, disable extra ray args
# export environment variables
# tos_region=""
# if [[ $TCE_ZONE =~ "North" ]]
# then
# tos_region="cn-north"
# fi

# export BYTED_RAY_REDIRECT_LOG="/var/log/tiger/ray/session_latest/logs"

# if ! [[ -z "$RAY_HISTORY_SERVER" ]]
# then
# export BYTED_RAY_HISTORY_SERVER_ENABLED=$RAY_HISTORY_SERVER
# export BYTED_RAY_HISTORY_SERVER_EVENT_LOG="tos://tos-$tos_region.byted.org/inf-batch-ray/history_server"
# fi

# export BYTED_RAY_CLUSTER="$CLOUDNATIVE_APPLICATION_ID-$RUNTIME_IDC_NAME-cloudnative"
# # BYTED_RAY_TOS_ACCESS_KEY will be injected by metis

### setup ray
HEAD_IP=$SERVER_IP
HEAD_PORT=$SERVER_PORT

ray_pod_ip=$HEAD_IP
if [[ $ray_pod_ip =~ "[" ]]
then
ipv6_prefix="["
ipv6_suffix="]"
ray_pod_ip=${ray_pod_ip#"$ipv6_prefix"}
ray_pod_ip=${ray_pod_ip%"$ipv6_suffix"}
fi
echo $ray_pod_ip
export BYTED_RAY_POD_IP=$ray_pod_ip

# if [[ $role != "head" ]]; then
#   # skip waiting head ready, retry
#   echo "worker executing: ulimit -n 65536; ray start --block --address=${HEAD_IP}:${HEAD_PORT} --node-ip-address=$node_addr --dashboard-agent-listen-port=$PORT0 --dashboard-agent-grpc-port=$PORT1 --num-cpus=${num_cpus} --num-gpus=${num_gpus} --memory=${memory} --object-store-memory=${obj_store_mem} --min-worker-port=0 --max-worker-port=0 --node-name=$MY_POD_NAME ${customized_resource}"
#   until ulimit -n 65536; ray start --block --address=${HEAD_IP}:${HEAD_PORT} --node-ip-address=$node_addr --dashboard-agent-listen-port=$PORT0 --dashboard-agent-grpc-port=$PORT1 --num-cpus=${num_cpus} --num-gpus=${num_gpus} --memory=${memory} --object-store-memory=${obj_store_mem} --min-worker-port=0 --max-worker-port=0 --node-name=$MY_POD_NAME ${customized_resource}
#   do
#     sleep 3s
#   done
#   echo "worker joined ray cluster"
# else
#   echo "head executing: ulimit -n 65536; ray start --head --block --node-ip-address=${HEAD_IP} --port=${HEAD_PORT} --dashboard-host='' --dashboard-port=$PORT1 --ray-client-server-port=$PORT2 --dashboard-agent-listen-port=$PORT3 --dashboard-agent-grpc-port=$PORT4 --node-name=$MY_POD_NAME --num-cpus=${num_cpus} --num-gpus=${num_gpus} --memory=${memory} --object-store-memory=${obj_store_mem} --min-worker-port=0 --max-worker-port=0 --plasma-directory=/dev/shm ${customized_resource}"
#   until ulimit -n 65536; ray start --head --block --node-ip-address=${HEAD_IP} --port=${HEAD_PORT} --dashboard-host='' --dashboard-port=$PORT1 --ray-client-server-port=$PORT2 --dashboard-agent-listen-port=$PORT3 --dashboard-agent-grpc-port=$PORT4 --node-name=$MY_POD_NAME --num-cpus=${num_cpus} --num-gpus=${num_gpus} --memory=${memory} --object-store-memory=${obj_store_mem} --min-worker-port=0 --max-worker-port=0 --plasma-directory=/dev/shm ${customized_resource}
#   do
#     sleep 3s
#   done
#   echo "ray head started"

#   ray job submit --address=${HEAD_IP}:${HEAD_PORT} --runtime-env-json='{
#         "working_dir": "/opt/tiger/ppo-long-cot/training/",
#         "py_modules": ["/opt/tiger/ppo-long-cot/training/"],
#         "pip": ["ray==2.12.0", "latex2sympy2", "timeout_decorator"]
#     }' -- /bin/bash ${RUN_SCRIPT}

# fi
ray start --head   --port=${HEAD_PORT} --dashboard-host='' --dashboard-port=$PORT1 --ray-client-server-port=$PORT2 --dashboard-agent-listen-port=$PORT3 --dashboard-agent-grpc-port=$PORT4 --node-name=$MY_POD_NAME --num-cpus=${num_cpus} --num-gpus=${num_gpus} --memory=${memory} --object-store-memory=${obj_store_mem} --min-worker-port=0 --max-worker-port=0 --plasma-directory=/dev/shm 
# if [[ $role != "head" ]]; then
#   # Worker node
 
#   # echo "worker executing: ulimit -n 65536; ray start --block --address=${HEAD_IP}:${HEAD_PORT} --node-ip-address=$node_addr --dashboard-agent-listen-port=$PORT0 --dashboard-agent-grpc-port=$PORT1 --num-cpus=${num_cpus} --num-gpus=${num_gpus} --memory=${memory} --object-store-memory=${obj_store_mem} --min-worker-port=0 --max-worker-port=0 --node-name=$MY_POD_NAME ${customized_resource}"
#   # until ulimit -n 65536; ray start --block --address=${HEAD_IP}:${HEAD_PORT} --node-ip-address=$node_addr --dashboard-agent-listen-port=$PORT0 --dashboard-agent-grpc-port=$PORT1 --num-cpus=${num_cpus} --num-gpus=${num_gpus} --memory=${memory} --object-store-memory=${obj_store_mem} --min-worker-port=0 --max-worker-port=0 --node-name=$MY_POD_NAME 
#   # do
#   #   sleep 3s
#   # done
#   ray start --block --address=${HEAD_IP}:${HEAD_PORT} --node-ip-address=$node_addr --dashboard-agent-listen-port=$PORT0 --dashboard-agent-grpc-port=$PORT1 --num-cpus=${num_cpus} --num-gpus=${num_gpus} --memory=${memory} --object-store-memory=${obj_store_mem} --min-worker-port=0 --max-worker-port=0 --node-name=$MY_POD_NAME 
#   echo "worker joined ray cluster"

# else
#   # Head node

#   echo "head executing: ulimit -n 65536; ray start --head --block --node-ip-address=${HEAD_IP} --port=${HEAD_PORT} --dashboard-host='' --dashboard-port=$PORT1 --ray-client-server-port=$PORT2 --dashboard-agent-listen-port=$PORT3 --dashboard-agent-grpc-port=$PORT4 --node-name=$MY_POD_NAME --num-cpus=${num_cpus} --num-gpus=${num_gpus} --memory=${memory} --object-store-memory=${obj_store_mem} --min-worker-port=0 --max-worker-port=0 --plasma-directory=/dev/shm ${customized_resource}"
#   # until ulimit -n 65536; ray start --head --block --node-ip-address=${HEAD_IP} --port=${HEAD_PORT} --dashboard-host='' --dashboard-port=$PORT1 --ray-client-server-port=$PORT2 --dashboard-agent-listen-port=$PORT3 --dashboard-agent-grpc-port=$PORT4 --node-name=$MY_POD_NAME --num-cpus=${num_cpus} --num-gpus=${num_gpus} --memory=${memory} --object-store-memory=${obj_store_mem} --min-worker-port=0 --max-worker-port=0 --plasma-directory=/dev/shm ${customized_resource}
#   # do
#   #   sleep 3s
#   # done
#   ray start --head  --node-ip-address=${HEAD_IP} --port=${HEAD_PORT} --dashboard-host='' --dashboard-port=$PORT1 --ray-client-server-port=$PORT2 --dashboard-agent-listen-port=$PORT3 --dashboard-agent-grpc-port=$PORT4 --node-name=$MY_POD_NAME --num-cpus=${num_cpus} --num-gpus=${num_gpus} --memory=${memory} --object-store-memory=${obj_store_mem} --min-worker-port=0 --max-worker-port=0 --plasma-directory=/dev/shm 
#   echo "ray head started"
set -x
CURRENT_DIR=$(pwd)
# Make the exclude path relative instead of absolute
DATA_EXCLUDE_PATH="/examples/simplelr_math_eval/data/"
# "working_dir": "'${CURRENT_DIR}'",
#          "env_vars": {
#             "http_proxy": "",
#             "https_proxy": ""
#          },
#           "excludes": ["/.git/", "'${DATA_EXCLUDE_PATH}'"]
# Submit job
ray job submit --address=127.0.0.1:${HEAD_PORT} \
  --runtime-env-json='{
        "working_dir": "'${CURRENT_DIR}'",
        "env_vars": {
          "http_proxy": "",
          "https_proxy": "",
          "RAY_DEBUG": "1",
          "WANDB_MODE": "offline"
        },
        "excludes": ["/.git/", "'${DATA_EXCLUDE_PATH}'"]
    }' \
      -- /bin/bash train_grpo_math_tune.sh \
        --model_name Qwen-2.5-0.5B \
        --suffix test_genrm_tiger_rm \
        --dataset_name simplelr_math_35  \
        --max_response_length 1000 \
        --train_batch_size 8 \
        --rollout_n 2 \
        --rollout_gpu_memory_util 0.75 \
        --rollout_tp 1  \
        --save_freq 1000 \
        --train_batch_size 32 \
        --ppo_mini_batch_size 8 \
        --genrm_enable True \
        --genrrm_prompt_type tiger-verifier \
        --genrrm_model_name tiger-verifier \
        --genrrm_temperature 0.0 \
        --genrrm_top_p 1.0 \
        --genrm_max_response_length 2048


# ray job submit --address=127.0.0.1:${HEAD_PORT} \
#   --runtime-env-json='{
#         "env_vars": {
#           "http_proxy": "",
#           "https_proxy": "",
#           "RAY_DEBUG": "1",
#           "WANDB_MODE": "offline"
#         }
#     }' \
#       -- /bin/bash train_grpo_math_tune.sh \
#         --model_name Qwen-2.5-0.5B \
#         --suffix test_genrm_tiger_rm \
#         --dataset_name simplelr_math_35  \
#         --max_response_length 1000 \
#         --train_batch_size 8 \
#         --rollout_n 2 \
#         --rollout_gpu_memory_util 0.75 \
#         --rollout_tp 1  \
#         --save_freq 1000 \
#         --train_batch_size 32 \
#         --ppo_mini_batch_size 8 \
#         --genrm_enable True \
#         --genrrm_prompt_type qwen-boxed_with_question \
#         --genrrm_model_name Qwen-2.5-1.5B-Instruct \
#         --genrrm_temperature 0.0 \
#         --genrrm_top_p 1.0 \
#         --genrm_max_response_length 2048

# ray job submit --address=127.0.0.1:${HEAD_PORT} \
#   --runtime-env-json='{
#         "env_vars": {
#           "http_proxy": "",
#           "https_proxy": ""
#         }
#     }' \
#       -- /bin/bash train_grpo_math_tune.sh \
#         --model_name Qwen-2.5-7B \
#         --suffix test_genrm \
#         --dataset_name simplelr_math_35 \
#         --max_response_length 8192 \
#         --train_batch_size 128 \
#         --rollout_n 8 \
#         --rollout_gpu_memory_util 0.75 \
#         --rollout_tp 2  \
#         --save_freq 1000 \
#         --genrrm_model_name DeepSeek-R1-Distill-Qwen-1.5B

# ray job submit --address=127.0.0.1:${HEAD_PORT} \
#   --runtime-env-json='{
#         "env_vars": {
#           "http_proxy": "",
#           "https_proxy": ""
#         }
#     }' \
#       -- /bin/bash train_grpo_math_tune.sh --model_name Qwen-2.5-7B --max_response_length 8192  --train_batch_size 256 --ppo_mini_batch_size 128 --rollout_n 8 --kl_loss_coef 0.0 --entropy_coeffient 0.0 --rollout_gpu_memory_util 0.65 --rollout_tp 2  --clip_ratio_high 0.28  --genrm_enable True --suffix test_genrm 

