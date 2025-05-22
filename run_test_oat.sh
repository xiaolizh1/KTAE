model_dirs=()
device=0
template="our"
temperatures="0.0"
max_tokens=8000
N_SAMPLING=1
top_p=1.0
use_chat=1
use_system_prompt=1
task="aime25,aime24,math500,amc23,minerva_math,olympiadbench"
echo "Model directories:"
for model_dir in "${model_dirs[@]}"; do
    echo "${model_dir}"
done
seeds="0"
for model_dir in "${model_dirs[@]}"; do
    CUDA_VISIBLE_DEVICES=${device} python eval_baseline.py \
        --model_name ${model_dir} \
        --template ${template} \
        --seeds ${seeds} \
        --temperatures ${temperatures} \
        --save True
done