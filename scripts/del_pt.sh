#!/bin/bash

model_dirs=()
project_name=()
experiment_name=()
checkpoints_name=()
for project in "${project_name[@]}"; do
    for experiment in "${experiment_name[@]}"; do
        for checkpoint in "${checkpoints_name[@]}"; do
            model_dirs+=("../ckpts/$project/$experiment/global_step_$checkpoint/actor")
        done
    done
done
for model_dir in "${model_dirs[@]}"; do
    if [ -d "$model_dir" ]; then
        find "$model_dir" -type f -name "*.pt" -exec rm -f {} \;
        echo "所有 .pt 文件已成功删除。"
    else
        echo "目录 $model_dir 不存在，请检查路径是否正确。"
    fi
done