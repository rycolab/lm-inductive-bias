set -euo pipefail
. experiments/include.bash

data_name="PFSA"
exp_names=("local_entropy")
BASE_DIR="$DATA_DIR"/"$data_name"
mem_per_cpu=16g

for exp_name in "${exp_names[@]}"; do
    for grammar_dir in "$BASE_DIR"/"$exp_name"/*; do
        grammar_name=$(basename "$grammar_dir")
        mkdir -p "$grammar_dir"/datasets/validation
        mkdir -p "$grammar_dir"/datasets/test
        if [ ! -f "$grammar_dir"/main.tok ]; then
            mv "$grammar_dir"/train.txt "$grammar_dir"/main.tok
        fi
        if [ ! -f "$grammar_dir"/datasets/validation/main.tok ]; then
            mv "$grammar_dir"/val.txt "$grammar_dir"/datasets/validation/main.tok
        fi
        if [ ! -f "$grammar_dir"/datasets/test/main.tok ]; then
            mv "$grammar_dir"/test.txt "$grammar_dir"/datasets/test/main.tok
        fi

        submit_job \
        prepare_data+"$data_name"+"$exp_name"+"$grammar_name" \
        cpu \
        --time=4:00:00 \
        --mem-per-cpu="$mem_per_cpu" \
        -- \
        python "$RAU_DIR"/src/rau/tasks/language_modeling/prepare_data.py \
            --training-data "$grammar_dir" \
            --more-data validation \
            --more-data test \
            --always-allow-unk
    done
done


