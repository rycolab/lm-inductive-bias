set -euo pipefail
. experiments/include.bash

data_dir="$DATA_DIR"/babylm2024_raw/test
mem_per_cpu=32g


for corpus_path in "$data_dir"/*; do
    corpus_name=$(basename "$corpus_path")
    dst_path="$data_dir"/"$corpus_name".txt

    submit_job \
        preprocess_babylm+"$corpus_name" \
        cpu \
        --mem-per-cpu="$mem_per_cpu" \
        --time=48:00:00 \
        --tasks 10 \
        -- \
        python data_preprocessing/preprocess_babylm.py \
        --input_file "$corpus_path" \
        --output_path "$dst_path" \
        --min-length 2 \
        --spacy-model en_core_web_lg \
        --n-jobs 10
done

