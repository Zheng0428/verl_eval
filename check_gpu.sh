#!/bin/bash
set -euo pipefail # 确保脚本在遇到错误时立即退出，并捕获管道中的错误

GPU_ID_TO_KILL=1 # 设置你想要操作的GPU编号

echo "--- 正在查找并杀死 GPU ${GPU_ID_TO_KILL} 上的进程 ---"

# 1. 使用 nvidia-smi 提取指定 GPU 上的进程 PID
#    -q -x: 输出XML格式的查询结果
#    xmllint --xpath: 使用XPath表达式从XML中精确提取PID
#    tr ' ' '\n': 将空格分隔的PID转换为每行一个，方便xargs处理
#    2>/dev/null: 抑制xmllint可能产生的错误信息
PIDS_ON_GPU=$(nvidia-smi -q -x | \
              xmllint --xpath "//gpu[index='${GPU_ID_TO_KILL}']/processes/process_info/pid/text()" - 2>/dev/null | \
              tr ' ' '\n')



# 2. 检查是否找到进程
if [ -z "$PIDS_ON_GPU" ]; then
    echo "在 GPU ${GPU_ID_TO_KILL} 上未找到任何正在运行的进程。"
else
    echo "在 GPU ${GPU_ID_TO_KILL} 上找到以下进程 (PIDs):"
    echo "$PIDS_ON_GPU"

    echo "正在尝试强制终止这些进程 (kill -9)..."
    # 3. 使用 xargs 将 PIDs 传递给 sudo kill -9 命令
    #    xargs -r: 如果没有输入，则不执行命令
    echo "$PIDS_ON_GPU" | xargs -r sudo kill -9
    
    echo "强制终止命令已发送。"
    
    # 4. 可选：短暂等待后再次检查，确认进程是否已终止
    sleep 2
    echo "等待 2 秒后再次检查 GPU ${GPU_ID_TO_KILL} 上的进程状态..."
    
    PIDS_AFTER_KILL=$(nvidia-smi -q -x | \
                      xmllint --xpath "//gpu[index='${GPU_ID_TO_KILL}']/processes/process_info/pid/text()" - 2>/dev/null | \
                      tr ' ' '\n')
    
    if [ -n "$PIDS_AFTER_KILL" ]; then
        echo "警告：GPU ${GPU_ID_TO_KILL} 上仍有进程残留！PIDs: $PIDS_AFTER_KILL"
    else
        echo "GPU ${GPU_ID_TO_KILL} 上的所有进程已成功清理。"
    fi
fi

echo "--- 操作完成 ---"