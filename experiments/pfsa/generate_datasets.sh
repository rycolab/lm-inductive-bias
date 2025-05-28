#!/bin/bash
set -euo pipefail
. experiments/include.bash

# Override default values if needed (uncomment for testing)
# N_SYMBOLS_LIST=(32)
# N_STATES_LIST=(16)
# N_TOPOLOGY_SEEDS=1
# N_WEIGHT_SEEDS=1

data_name="PFSA"
exp_name="local_entropy_XXX"
output_dir="$DATA_DIR/$data_name/$exp_name"
mem_per_cpu=16g

# Ensure the output directory exists
mkdir -p "$output_dir"
mkdir -p logs

# Loop over all combinations of n_symbols and n_states
for n_symbols in "${N_SYMBOLS_LIST[@]}"; do
    for n_states in "${N_STATES_LIST[@]}"; do
        for topology_seed in $(seq 1 $N_TOPOLOGY_SEEDS); do
            for weight_seed in $(seq 1 $N_WEIGHT_SEEDS); do
                job_name="generate_pfsa_${n_symbols}_${n_states}_${topology_seed}_${weight_seed}_${exp_name}"

                submit_job \
                    "$job_name" \
                    cpu \
                    --time=24:00:00 \
                    --mem-per-cpu="$mem_per_cpu" \
                    --cpus-per-task=1 \
                    -- \
                    python pfsa/sample_dataset.py \
                        n_symbols=$n_symbols \
                        n_states=$n_states \
                        topology_seed=$topology_seed \
                        weight_seed=$weight_seed \
                        output=$output_dir
            done
        done
    done
done
