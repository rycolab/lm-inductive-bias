set -euo pipefail
. experiments/include.bash

echo "${ARCHITECTURES[@]}"
data_name="PFSA"
exp_names=("local_entropy")
exp_base_dir="$DATA_DIR"/"$data_name"
split_names=("test" "validation")


if [ "$data_name" == "PFSA" ]; then
    metadata_filename="metadata.json"
fi

for split_name in "${split_names[@]}"; do
    for exp_name in "${exp_names[@]}"; do
        if [ "$data_name" == "PFSA" ]; then
            OUTPUT_PATH="${RESULTS_DIR}"/"$data_name"/"$exp_name"/collected_results_"$split_name".csv
        else
            OUTPUT_PATH="${RESULTS_DIR}"/"$data_name"/"$exp_name"/collected_results_"$ngram_method"_"$split_name".csv
        fi


        # submit_job \
        #     collect_result+"$data_name"+"$exp_name"+"$split_name" \
        #     cpu \
        #     --time=4:00:00 \
        #     -- \
            python src/analysis/collect_results.py \
            --data_dir "$exp_base_dir" \
            --result_base_dir "$RESULTS_DIR"/"$data_name" \
            --exp_name "$exp_name" \
            --architectures "${ARCHITECTURES[@]}" \
            --split_name "$split_name" \
            --output_path "$OUTPUT_PATH" \
            --metadata_filename "$metadata_filename"
    done
done
