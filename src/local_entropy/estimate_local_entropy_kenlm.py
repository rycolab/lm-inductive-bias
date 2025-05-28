import gzip
import math
import kenlm
import argparse
import os
import json
import subprocess
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count


def load_data(train_path: Path, valid_path: Path, test_path: Path) -> list:
    """
    Load sentences from specified files for train, validation, and test.
    Each line is treated as a sentence.
    Returns a list of all sentences.
    """
    all_sentences = []

    paths = [path for path in [train_path, valid_path, test_path] if path.exists()]

    for path in paths:
        with path.open("r") as f:
            sentences = f.read().splitlines()
            all_sentences.extend(sentences)

    return all_sentences


def calculate_entropy(model, text):
    log_prob_sum = 0
    word_count = 0

    for line in text:
        log_prob_sum += model.score(line, bos=True, eos=True) * math.log2(10)
        word_count += len(line.split()) + 1

    return -1 * (log_prob_sum / word_count)


def caculate_mlocal_entropy(model, text, n: int):
    total_local_entropy = 0
    denominator = 0

    for line in text:
        scores = list(model.full_scores(line))
        valid_scores = scores[n - 1 :]
        if len(valid_scores) == 0:
            continue
        assert len(valid_scores) == len(line.split()) - n + 2, (
            f"{len(valid_scores)} != {len(line.split()) - n + 2}"
        )
        for prob, _, _ in valid_scores:
            local_entropy = -prob * math.log2(10)
            total_local_entropy += local_entropy
            denominator += 1

    return total_local_entropy / denominator if denominator > 0 else float("inf")


def process_single_n(args):
    work_file_path, n, lmplz_path, sentences, memory, method = args
    arpa_path = work_file_path.with_suffix(f".{n}.arpa")

    try:
        with open(work_file_path, "w") as f:
            for line in sentences:
                f.write(line + "\n")

        subprocess.run(
            [
                str(lmplz_path),
                "-o",
                str(n),
                "--skip_symbols",
                "--discount_fallback",
                "--memory",
                memory,
                "--text",
                str(work_file_path),
                "--arpa",
                str(arpa_path),
            ],
            check=True,
        )

        model = kenlm.Model(str(arpa_path))
        if method == "entropy":
            entropy = calculate_entropy(model, sentences)
        elif method == "mlocal_entropy":
            entropy = caculate_mlocal_entropy(model, sentences, n)
        else:
            raise ValueError(f"Invalid method: {method}")
        return n, entropy

    except Exception as e:
        print(f"Error processing {n}-gram: {e}")
        arpa_path.unlink(missing_ok=True)
        return n, None


def main():
    parser = argparse.ArgumentParser(description="Calculate n-gram entropy using KenLM")
    parser.add_argument(
        "--train_path", type=Path, required=True, help="Path to training data file"
    )
    parser.add_argument(
        "--valid_path", type=Path, required=True, help="Path to validation data file"
    )
    parser.add_argument(
        "--test_path", type=Path, required=True, help="Path to test data file"
    )
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5],
        help="n-gram sizes (default: 2 3 4 5)",
    )
    parser.add_argument(
        "--kenlm-path", type=str, default="../kenlm", help="Path to KenLM directory"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Number of parallel processes (default: number of CPU cores)",
    )
    parser.add_argument(
        "--memory", type=str, default="8G", help="Memory limit for KenLM (default: 4G)"
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="work",
        help="Working directory for intermediate files",
    )
    parser.add_argument(
        "--output_path", type=Path, required=True, help="Path to output JSON file"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="mlocal_entropy",
        help="Method to calculate mlocal_entropy (default: mlocal_entropy)",
    )
    args = parser.parse_args()

    print("Loading datasets...")
    sentences = load_data(args.train_path, args.valid_path, args.test_path)

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    work_file_path = work_dir / f"{args.train_path.name}.txt"

    kenlm_build_dir = Path(args.kenlm_path) / "build"
    lmplz_path = kenlm_build_dir / "bin" / "lmplz"

    if not lmplz_path.exists():
        raise FileNotFoundError(f"lmplz not found at {lmplz_path}")

    num_processes = args.num_processes or cpu_count()
    print(f"Using {num_processes} processes")

    process_args = [
        (work_file_path, n, lmplz_path, sentences, args.memory, args.method)
        for n in args.n
    ]
    with Pool(num_processes) as pool:
        model_results = pool.map(process_single_n, process_args)

    final_results = {"local_entropy": {}}
    print(model_results)
    for n, entropy in model_results:
        if entropy is not None:
            final_results["local_entropy"][str(n)] = entropy

    with open(args.output_path, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"\nResults saved to {args.output_path}")

    if work_file_path.exists():
        work_file_path.unlink()


if __name__ == "__main__":
    main()
