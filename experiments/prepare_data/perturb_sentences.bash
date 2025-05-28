set -euo pipefail
. experiments/include.bash

exp_name="deterministic_shuffles"
data_name="BLLIP_SM"
EXP_DIR="$DATA_DIR"/"$data_name"/"$exp_name"
BASE_DIR="$EXP_DIR"/Base


submit_job \
    "perturb_sentences+${data_name}+${exp_name} \
    cpu \
    --tasks=48 \
    --mem-per-cpu=8g \
    --time=8:00:00 \
    -- \
    python perturbation/perturb_sentences.py \
    --base_train_file "$BASE_DIR"/train.txt \
    --base_dev_file "$BASE_DIR"/dev.txt \
    --base_test_file "$BASE_DIR"/test.txt \
    --exp_dir "$EXP_DIR" \
    --perturb_config_file $PERTURBATION_CONFIG_FILE \
    --n_workers 48



