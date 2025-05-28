set -euo pipefail

# Get arguments
data_dir="$1"
output_dir="$2"
examples_per_checkpoint="$3"
max_tokens_per_batch="$4"

python "$RAU_DIR"/src/rau/tasks/language_modeling/train.py \
    --training-data "$data_dir" \
    --architecture transformer \
    --num-layers 4 \
    --d-model 768 \
    --num-heads 12 \
    --feedforward-size 3072 \
    --dropout 0.1 \
    --init-scale 0.1 \
    --max-epochs 1000 \
    --max-tokens-per-batch "$max_tokens_per_batch" \
    --optimizer Adam \
    --initial-learning-rate 0.0005 \
    --gradient-clipping-threshold 5 \
    --early-stopping-patience 10 \
    --learning-rate-patience 5 \
    --learning-rate-decay-factor 0.5 \
    --examples-per-checkpoint "$examples_per_checkpoint" \
    --output "$output_dir"

eval_dir="$output_dir"/evaluation
mkdir -p "$eval_dir"

python "$RAU_DIR"/src/rau/tasks/language_modeling/evaluate.py \
    --load-model "$output_dir" \
    --training-data "$data_dir" \
    --input test \
    --batching-max-tokens 2048 > "$eval_dir"/test.json

python "$RAU_DIR"/src/rau/tasks/language_modeling/evaluate.py \
    --load-model "$output_dir" \
    --training-data "$data_dir" \
    --input validation \
    --batching-max-tokens 2048 > "$eval_dir"/validation.json

python evaluate/add_base2_metrics.py \
    --input_path "$eval_dir"/test.json

python evaluate/add_base2_metrics.py \
    --input_path "$eval_dir"/validation.json
