#!/bin/bash
set -euo pipefail
# . experiments/include.bash

DATA_DIR=
data_name="PFSA"
BASE_DIR="$DATA_DIR"/"$data_name"
exp_name="local_entropy_rebuttal"

for grammar_dir in "$BASE_DIR"/"$exp_name"/*; do
    grammar_name=$(basename "$grammar_dir")


    train_file="$grammar_dir/train.txt"
    valid_file="$grammar_dir/val.txt"
    test_file="$grammar_dir/test.txt"
    output_file="$grammar_dir/metadata_kenlm_mlocal_entropy.json"


    if [[ ! -f "$train_file" || ! -f "$valid_file" || ! -f "$test_file" ]]; then
        echo "Required files not found in $grammar_dir. Skipping..."
        continue
    fi

    python src/local_entropy/estimate_local_entropy_kenlm.py \
        --train_path "$train_file" \
        --valid_path "$valid_file" \
        --test_path "$test_file" \
        --output_path "$output_file" \
        --n 2 3 4 5 \
        --memory 8G \
        --num-processes 1 \
        --method mlocal_entropy \
        --work-dir "work/${data_name}_${exp_name}_${grammar_name}"
done




