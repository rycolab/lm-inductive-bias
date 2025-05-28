from hydra.utils import instantiate
import argparse
from pathlib import Path
import gzip
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import json
import logging


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def process_sentence(perturb_func, sentence):
    return perturb_func.perturb(sentence)


def load_sentences(file_path):
    """Load sentences from a file (gzipped or plain text)"""
    if str(file_path).endswith(".gz"):
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            return f.read().splitlines()
    else:
        with open(file_path, "rt", encoding="utf-8") as f:
            return f.read().splitlines()


def save_sentences(sentences, output_file):
    """Save sentences to a plain text file"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wt", encoding="utf-8") as f:
        f.write("\n".join(sentences))


def process_split(perturb_func, input_file, output_file, n_workers, split_name, logger):
    """Process a single data split"""
    logger.info(f"Processing {split_name} split from {input_file}")
    sentences = load_sentences(input_file)

    with mp.Pool(processes=n_workers) as pool:
        process_func = partial(process_sentence, perturb_func)
        perturbed_sentences = list(
            tqdm(
                pool.imap(process_func, sentences),
                total=len(sentences),
                desc=f"Perturbing {split_name}",
            )
        )

    logger.info(f"Saving {len(perturbed_sentences)} sentences to {output_file}")
    save_sentences(perturbed_sentences, output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_train_file", type=lambda p: Path(p).resolve(), required=True
    )
    parser.add_argument(
        "--base_dev_file", type=lambda p: Path(p).resolve(), required=True
    )
    parser.add_argument(
        "--base_test_file", type=lambda p: Path(p).resolve(), required=True
    )
    parser.add_argument("--exp_dir", type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument(
        "--perturb_config_file", type=lambda p: Path(p).resolve(), required=True
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=mp.cpu_count(),
        help="Number of worker processes (default: number of CPU cores)",
    )
    args = parser.parse_args()

    logger = setup_logger()

    # Load perturbation configurations
    with open(args.perturb_config_file, "r") as f:
        perturb_config = json.load(f)

    logger.info(f"Found {len(perturb_config)} perturbation configurations:")
    for k, v in perturb_config.items():
        logger.info(f"{k}: {v}")

    # Process each perturbation type
    for perturb_name, perturb_params in perturb_config.items():
        logger.info(f"\nProcessing perturbation: {perturb_name}")
        perturb_func = instantiate(perturb_params)

        # Create output directory for this perturbation
        perturb_dir = args.exp_dir / f"{perturb_name}"

        # Process each split
        splits = {
            "train": (args.base_train_file, perturb_dir / "train.txt"),
            "dev": (args.base_dev_file, perturb_dir / "dev.txt"),
            "test": (args.base_test_file, perturb_dir / "test.txt"),
        }

        for split_name, (input_file, output_file) in splits.items():
            process_split(
                perturb_func=perturb_func,
                input_file=input_file,
                output_file=output_file,
                n_workers=args.n_workers,
                split_name=split_name,
                logger=logger,
            )


if __name__ == "__main__":
    main()
