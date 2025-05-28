set -euo pipefail
. experiments/include.bash

data_name="PFSA"
exp_names=("local_entropy_XXX_only")
BASE_DIR="$DATA_DIR"/"$data_name"

for exp_name in "${exp_names[@]}"; do
    for grammar_dir in "$BASE_DIR"/"$exp_name"/*; do
        # Extract the number to remove from directory name
        dir_name=$(basename "$grammar_dir")
        number_to_remove=$(echo "$dir_name" | grep -oP 'Q\d+_S\K(\d+)' || echo "")
        echo "dir_name: $dir_name"
        echo "number_to_remove: $number_to_remove"

        if [ -n "$number_to_remove" ]; then
            for target in "train.txt" "val.txt" "test.txt"; do
                target_file="$grammar_dir"/"$target"
                sed -i -E "s/(^|[[:space:]])$number_to_remove$//g" "$target_file"
            done
        fi
    done
done
