set -euo pipefail
. experiments/include.bash

data_name="PFSA"
exp_names=("local_entropy")
exp_base_dir="$DATA_DIR"/"$data_name"
examples_per_checkpoint=10000
max_tokens_per_batch=2048
time_limit=04:00:00
gpu_mem=10g

for exp_name in "${exp_names[@]}"; do
    for trial in $(seq 0 $(($NUM_TRIALS - 1))); do
        for data_dir in "$exp_base_dir"/"$exp_name"/*; do
            grammar_name=$(basename "$data_dir")
            if [ -f "$RESULTS_DIR"/"$data_name"/"$exp_name"/lstm/"$grammar_name"_trial"$trial"/evaluation/validation.json ]; then
                continue
            fi

            submit_job \
            train_lstm+"$data_name"+"$exp_name"+"$grammar_name"+trial"$trial" \
            gpu \
            --gpus=1 \
            --gres=gpumem:$gpu_mem \
            --mem-per-cpu=16g \
            --tmp=20g \
            --time="$time_limit" \
            -- \
            bash neural_networks/train_and_evaluate_lstm.sh \
                "$data_dir" \
                "$RESULTS_DIR"/"$data_name"/"$exp_name"/lstm/"$grammar_name"_trial"$trial" \
                "$examples_per_checkpoint" \
                "$max_tokens_per_batch"
        done
    done
done


