from pathlib import Path
import json
import math
import argparse


def add_base2_metrics(input_path: Path):
    with open(input_path, "r") as f:
        data = json.load(f)

    data["cross_entropy_per_token_base_e"] = data["cross_entropy_per_token"]
    data["cross_entropy_per_token_base_2"] = data["cross_entropy_per_token"] / math.log(
        2
    )

    with open(input_path, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, required=True)
    args = parser.parse_args()

    add_base2_metrics(args.input_path)
    print(f"Added base2 metrics to {args.input_path}")
