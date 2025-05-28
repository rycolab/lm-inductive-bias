import json
import os
import random
from collections import Counter
from math import ceil
from typing import List, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig

from fsa import PFSA
from fsa_generator import random_dpfsa, random_ngram


def _save_dataset(D: List[Tuple[str, float]], fname: str) -> None:

    with open(fname, "w") as f:
        for ii, y in enumerate(D):
            if ii == len(D) - 1:
                f.write(y)
            else:
                f.write(y + "\n")


def save_dataset(
    D_train: List[Tuple[str, float]],
    D_val: List[Tuple[str, float]],
    D_test: List[Tuple[str, float]],
    data_dir: str,
) -> None:

    _save_dataset(D_train, os.path.join(data_dir, "train.txt"))
    _save_dataset(D_val, os.path.join(data_dir, "val.txt"))
    _save_dataset(D_test, os.path.join(data_dir, "test.txt"))

    print("Saved the datasets.")


def save_metadata(
    A: PFSA,
    data_dir: str,
    n_states: int,
    N_sym: int,
    N_train: int,
    N_val: int,
    N_test: int,
    m_max: int,
    topology_seed: int,
    weight_seed: int,
) -> None:
    model_path = os.path.join(data_dir, "model.pickle")

    metadata = {
        "fname": model_path,
        "n_states": n_states,
        "N_sym": N_sym,
        "N_train": N_train,
        "N_val": N_val,
        "N_test": N_test,
        "topology_seed": topology_seed,
        "weight_seed": weight_seed,
        "mean_length": float(A.mean_length),
        "entropy": float(A.entropy),
        "next_symbol_entropy": float(A.next_symbol_entropy),
        "local_entropy": {m: float(A.local_entropy(m)) for m in range(2, m_max + 1)},
    }

    json.dump(metadata, open(os.path.join(data_dir, "metadata.json"), "w"))

    A.save(model_path)

    print("Saved the metadata.")


def get_disjoint_data(
    A: PFSA, N_train: int, N_val: int, N_test: int, topology_seed: int
) -> Tuple[List[str], List[str], List[str]]:

    N_train_ = ceil(N_train * 1.1)
    N_val_ = ceil(N_val * 1.1)
    N_test_ = ceil(N_test * 1.1)

    random.seed(topology_seed)

    N = N_train_ + N_val_ + N_test_
    all_strings = A.sample(2 * N, logp=False, to_string=True)

    string_counts = Counter(all_strings)
    D = sorted(string_counts.keys(), key=lambda x: -string_counts[x])
    cs = np.cumsum([string_counts[x] for x in D])
    up_to = np.searchsorted(cs, N)
    D = D[:up_to]
    random.shuffle(D)

    all_strings = [x for x in all_strings if x in D]
    string_counts = Counter(all_strings)
    cs_ = np.cumsum([string_counts[x] for x in D])
    up_to_train = np.searchsorted(cs_, N_train_) + 1
    up_to_val = np.searchsorted(cs_, N_train_ + N_val_) + 1
    train_strings = D[:up_to_train]
    val_strings = D[up_to_train:up_to_val]
    test_strings = D[up_to_val:]

    D_train = [x for x in all_strings if x in train_strings][:N_train_]
    D_val = [x for x in all_strings if x in val_strings][:N_test_]
    D_test = [x for x in all_strings if x in test_strings][:N_test_]

    random.shuffle(D_train)
    random.shuffle(D_val)
    random.shuffle(D_test)

    D_train = D_train[:N_train]
    D_val = D_val[:N_val]
    D_test = D_test[:N_test]

    return D_train, D_val, D_test


def get_data(
    A: PFSA, N_train: int, N_val: int, N_test: int, disjoint: bool, topology_seed: int
) -> Tuple[List[str], List[str], List[str]]:

    print("Generating data...")

    if disjoint:
        D_train, D_val, D_test = get_disjoint_data(
            A, N_train, N_val, N_test, topology_seed
        )
    else:
        D_train = A.sample(N_train, logp=False, to_string=True)
        D_val = A.sample(N_val, logp=False, to_string=True)
        D_test = A.sample(N_test, logp=False, to_string=True)

    print("Generated the data.")
    print(f"Train: {len(D_train)}, Val: {len(D_val)}, Test: {len(D_test)}")
    print(f"ML(D_train) = {sum(len(x) for x in D_train) / (2 * len(D_train))}")
    print(f"ML(D_val) = {sum(len(x) for x in D_val) / (2 * len(D_val))}")
    print(f"ML(D_test) = {sum(len(x) for x in D_test) / (2 * len(D_test))}")

    return D_train, D_val, D_test


@hydra.main(version_base=None, config_path="../config", config_name="pfsa_config.yaml")
def generate_dataset(cfg: DictConfig):

    n_states = cfg.n_states
    n_symbols = cfg.n_symbols
    mean_length = cfg.mean_length
    N_train = cfg.N_train
    N_val = cfg.N_val
    N_test = cfg.N_test
    disjoint = cfg.disjoint
    topology_seed = cfg.topology_seed
    weight_seed = cfg.weight_seed
    m_max = cfg.m_max
    base_dir = cfg.output

    print(
        f"Generating dataset with n_states = {n_states}, n_symbols = {n_symbols}, "
        f"N_train = {N_train}, N_val = {N_val}, N_test = {N_test}, disjoint = {disjoint}, "
        f"topology_seed = {topology_seed}, weight_seed = {weight_seed}"
    )

    A = random_dpfsa(
        n_states,
        n_symbols,
        conditions=[lambda A: mean_length - 5 < A.mean_length < mean_length + 5],
        mean_length=mean_length,
        topology_seed=topology_seed,
        weight_seed=weight_seed,
    )

    print("Generated the random DPFSA.")
    D_train, D_val, D_test = get_data(
        A, N_train, N_val, N_test, disjoint, topology_seed
    )

    data_dir = os.path.join(
        base_dir, f"Q{n_states}_S{n_symbols}_ts{topology_seed}_ws{weight_seed}"
    )

    os.makedirs(data_dir, exist_ok=True)
    save_dataset(D_train, D_val, D_test, data_dir)

    save_metadata(
        A,
        data_dir,
        n_states,
        n_symbols,
        N_train,
        N_val,
        N_test,
        m_max,
        topology_seed,
        weight_seed,
    )


if __name__ == "__main__":
    generate_dataset()
