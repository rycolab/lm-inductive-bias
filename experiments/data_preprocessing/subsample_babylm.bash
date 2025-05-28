set -euo pipefail
. experiments/include.bash

src_dir="$DATA_DIR"/babylm2024_10M/deterministic_shuffles/Base
dst_dir="$DATA_DIR"/babylm2024_100K_sents/deterministic_shuffles/Base
train_sample_size=100000
dev_sample_size=10000
test_sample_size=10000

targets=("main.tok" "datasets/validation/main.tok" "datasets/test/main.tok")

for target in "${targets[@]}"; do
    src_path="$src_dir"/"$target"


    # Set sample size according to file type
    if [[ "$target" == "main.tok" ]]; then
        sample_size=$train_sample_size
        dst_path="$dst_dir"/train.txt
    elif [[ "$target" == *"validation"* ]]; then
        sample_size=$dev_sample_size
        dst_path="$dst_dir"/dev.txt
    else
        sample_size=$test_sample_size
        dst_path="$dst_dir"/test.txt
    fi

    mkdir -p "$(dirname "$dst_path")"
    echo "sample_size: $sample_size"
    echo "dst_path: $dst_path"

    # Shuffle and sample
    shuf -n "$sample_size" "$src_path" > "$dst_path"
done
