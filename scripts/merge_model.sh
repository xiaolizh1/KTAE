#!/bin/bash
model_dirs=()
project_name=()
experiment_name=()
checkpoints_name=()
for project in "${project_name[@]}"; do
    for experiment in "${experiment_name[@]}"; do
        for checkpoint in "${checkpoints_name[@]}"; do
            model_dirs+=("../checkpoints/$project/$experiment/global_step_$checkpoint/actor")
        done
    done
done

for model_dir in "${model_dirs[@]}"; do
    echo "Merging model at global step $model_dir"
    python model_merger.py --local_dir "$model_dir"
done