SINGULARITY_IMAGE_FILE=
DATA_DIR=
RESULTS_DIR=
RAU_DIR=

FIGURES_DIR=$RESULTS_DIR/figures
PERTURBATION_CONFIG_FILE=config/perturbation_func.json
ARCHITECTURES=("lstm" "transformer")
ARCHITECTURE_LABELS=("LSTM" "Transformer")

NUM_TRIALS=5

# PFSA dataset generation parameters
N_SYMBOLS_LIST=(32 48 64)
N_STATES_LIST=(16 24 32)
N_TOPOLOGY_SEEDS=5
N_WEIGHT_SEEDS=5

submit_job() {
  bash experiments/submit_job.bash "$@"
}
