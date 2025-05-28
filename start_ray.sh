#!/bin/bash
source setup_env.sh

USER_ENV=`whoami`
export ARNOLD_BYTEDRAY_ray_io_param_head_no_cpu=1
export RAY_LOG_DIR="/opt/tiger/ray"
export NCCL_DEBUG=DEBUG
export RAY_BACKEND_LOG_LEVEL=debug
export RAY_DEDUP_LOGS=1
### prepare basic env
### we can asssume in ARNOLD_MIXED_RAY mode, only supposed role will execute this script
role=$ARNOLD_ROLE
role_uppercase=${role^^}

### prepare configs for ray start
server_hosts=${ARNOLD_HEAD_HOSTS}
if [[ $role == "head" ]]; then
  worker_id=-1
  IFS=':' SERVER_CONFIGS=($server_hosts)
  unset IFS
  num_cpus=${ARNOLD_HEAD_CPU} 
  num_gpus=${ARNOLD_HEAD_GPU} 
  num_memory=$(($ARNOLD_HEAD_MEM >> 10))
  memory=$(($ARNOLD_HEAD_MEM * 1024 ** 2))
  obj_store_mem_limit=$(($ARNOLD_HEAD_MEM * 450 * 1024))
  if [[ "$ARNOLD_HEAD_GPU" -eq "0" ]]; then
    obj_store_mem=$((64 * 1024 ** 3))
  else
    obj_store_mem=$((${ARNOLD_HEAD_GPU:-1} * 64 * 1024 ** 3))
  fi
else
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
  fi
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

RUN_SCRIPT=$1
shift

if [[ $role != "head" ]]; then
  while true; do
      if ray status --address=${HEAD_IP}:${HEAD_PORT} &>/dev/null; then
          echo "Found Ray head node"
          break
      fi
  echo "$worker_id wait 5s for detecting ray main node initialization"
  sleep 5s
  done
  echo "Using number of cpus: ${num_cpus}, number of gpus: ${num_gpus}, memory: ${memory}, object store memory: ${obj_store_mem}"
  ray start --block --address=${HEAD_IP}:${HEAD_PORT} --node-ip-address=$node_addr --dashboard-agent-listen-port=$PORT0 --dashboard-agent-grpc-port=$PORT1 --num-cpus=${num_cpus} --num-gpus=${num_gpus} --memory=${memory} --object-store-memory=${obj_store_mem} --min-worker-port=0 --max-worker-port=0 --node-name=$MY_POD_NAME 
  echo "worker joined ray cluster"

else
  echo "head executing: ulimit -n 65536; ray start --head --block --node-ip-address=${HEAD_IP} --port=${HEAD_PORT} --dashboard-host='' --dashboard-port=$PORT1 --ray-client-server-port=$PORT2 --dashboard-agent-listen-port=$PORT3 --dashboard-agent-grpc-port=$PORT4 --node-name=$MY_POD_NAME --num-cpus=${num_cpus} --num-gpus=${num_gpus} --memory=${memory} --object-store-memory=${obj_store_mem} --min-worker-port=0 --max-worker-port=0 --plasma-directory=/dev/shm ${customized_resource}"
  ray start --head  --node-ip-address=${HEAD_IP} --port=${HEAD_PORT} --dashboard-host='' --dashboard-port=$PORT1 --ray-client-server-port=$PORT2 --dashboard-agent-listen-port=$PORT3 --dashboard-agent-grpc-port=$PORT4 --node-name=$MY_POD_NAME --num-cpus=0 --num-gpus=${num_gpus} --memory=${memory} --object-store-memory=${obj_store_mem} --min-worker-port=0 --max-worker-port=0 --plasma-directory=/dev/shm 
  echo "ray head started"
  set -x
  # Get current working directory for dynamic path resolution
  CURRENT_DIR=$(pwd)
  DATA_EXCLUDE_PATH="/examples/simplelr_math_eval/data/" # here, consider the root as current directory
  
  ray job submit --address=${HEAD_IP}:${HEAD_PORT} \
    --entrypoint-num-cpus=1 \
    --runtime-env-json='{
         "working_dir": "'${CURRENT_DIR}'",
         "env_vars": {
            "http_proxy": "",
            "https_proxy": ""
         },
          "excludes": ["/.git/", "'${DATA_EXCLUDE_PATH}'"]
      }' \
    -- /bin/bash $RUN_SCRIPT "$@"
fi