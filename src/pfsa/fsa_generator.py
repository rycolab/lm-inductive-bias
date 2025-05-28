from itertools import product
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from fsa import PFSA
from ngram import NGram


def random_pfsa(n_states: int, n_symbols: int, seed: int) -> PFSA:
    rng = np.random.default_rng(seed)
    A = PFSA(n_states, n_symbols)
    A.λ = rng.uniform(0, 1, n_states)
    A.λ /= A.λ.sum()
    A.ρ = rng.uniform(0, 1, n_states)
    A.Ts = {y: rng.uniform(0, 1, (n_states, n_states)) for y in range(n_symbols)}
    for q in range(A.n_states):
        s = sum(A.Ts[y][q].sum() for y in range(A.n_symbols)) + A.ρ[q]
        for y in range(A.n_symbols):
            A.Ts[y][q] /= s
        A.ρ[q] /= s

    return A


def _generate_outgoing_labels(
    n_states: int, n_symbols: int, rng: Optional[np.random.Generator], min_size: int = 2
):
    numbers = np.arange(n_symbols)
    sets = [set() for _ in range(n_states)]

    # Step 1: Ensure each state gets at least two unique symbols
    for state in range(n_states):
        sets[state].update(
            rng.choice(numbers, size=min(min_size, n_symbols), replace=False)
        )

    # Step 2: Ensure each symbol appears in at least one set
    for num in numbers:
        chosen_set = rng.choice(sets)
        chosen_set.add(num)

    # Step 3: Add additional symbols to create overlap
    total_size = sum(
        rng.integers(0, max(1, n_symbols // 2 - 2)) for _ in range(n_states)
    )
    for _ in range(total_size):
        num = rng.choice(numbers)
        chosen_set = rng.choice(sets)
        chosen_set.add(num)

    return sets


def _random_dpfsa(
    n_states: int,
    n_symbols: int,
    mean_length: Optional[float],
    topology_rng: np.random.Generator,
    weight_rng: np.random.Generator,
) -> PFSA:
    A = PFSA(n_states, n_symbols)
    qI = topology_rng.choice(n_states, 1)
    A.λ = np.zeros(n_states)
    A.λ[qI] = 1
    A.Ts = {y: np.zeros((A.n_states, A.n_states)) for y in range(A.n_symbols)}
    not_used = set(range(A.n_states))
    out_arcs = _generate_outgoing_labels(n_states, n_symbols, topology_rng)

    for q in range(A.n_states):
        # for y in out_arcs[q]:
        for y in range(A.n_symbols):
            if len(not_used) == 0:
                t = topology_rng.choice(n_states, 1)
            else:
                t = topology_rng.choice(list(not_used), 1)
                not_used.remove(t[0])
            A.Ts[y][q, t] = weight_rng.exponential(0.1) * int(y in out_arcs[q]) + 0.001

    for q in range(A.n_states):
        t = sum(A.Ts[y][q].sum() for y in range(A.n_symbols))
        if mean_length is None:
            A.ρ[q] = weight_rng.exponential(t / 25)
        else:
            A.ρ[q] = t / mean_length
            # A.ρ[q] = weight_rng.uniform(
            #     t / mean_length - 0.001, t / mean_length + 0.001
            # )
        s = t + A.ρ[q]
        for y in range(A.n_symbols):
            A.Ts[y][q] /= s
        A.ρ[q] /= s

    return A


def random_dpfsa(
    n_states: int,
    n_symbols: int,
    conditions: Sequence[Callable[[PFSA], bool]],
    topology_seed: int,
    weight_seed: int,
    mean_length: Optional[float] = None,
) -> PFSA:
    topology_rng = np.random.default_rng(topology_seed)
    weight_rng = np.random.default_rng(weight_seed)
    A = _random_dpfsa(n_states, n_symbols, mean_length, topology_rng, weight_rng)

    while (
        np.linalg.cond(np.eye(A.n_states) - A.T) > 1e3
        or not (1e-2 < np.linalg.norm(A.kleene) < 1e3)
        or not all(condition(A) for condition in conditions)
    ):
        A = _random_dpfsa(n_states, n_symbols, mean_length, topology_rng, weight_rng)

        topology_seed = topology_rng.integers(0, 2**32)
        topology_rng = np.random.default_rng(topology_seed)

    return A


def _next_symbol_probabilities(
    alphabet: Sequence[int],
    EOS: int,
    BOS: int,
    μ: float,
    rng: np.random.Generator,
) -> np.ndarray:

    p = rng.dirichlet((0.1,) * (len(alphabet) + 1))
    p[EOS] = 0
    p = p / np.sum(p)
    p[EOS] = 1 / (μ - 1)
    p = p / (1 + 1 / (μ - 1))

    return p


def random_ngram(n_symbols: int, n: int, seed: int, weighted: bool = True) -> PFSA:
    rng = np.random.default_rng(seed)
    A = NGram(n_symbols, n)

    EOS, BOS = n_symbols, n_symbols + 1
    alphabet = range(n_symbols)

    ngram2state: Dict[Tuple[int, ...], int] = {}
    p: Dict[int, np.ndarray] = {}

    # pre-pad with BOS
    ngram = tuple((BOS,) * (n - 1))
    idx = len(ngram2state)
    if weighted:
        p[idx] = _next_symbol_probabilities(alphabet, EOS, BOS, 10, rng)
    else:
        p[idx] = np.ones(n_symbols + 1)
    ngram2state[ngram] = idx

    # pre-pad with BOS
    for ll in range(n - 2, 0, -1):
        for ngr in product(alphabet, repeat=ll):
            ngram = tuple((BOS,) * (n - ll - 1) + ngr)
            idx = len(ngram2state)
            if weighted:
                p[idx] = _next_symbol_probabilities(alphabet, EOS, BOS, 10, rng)
            else:
                p[idx] = np.ones(n_symbols + 1)
            ngram2state[ngram] = idx

    # loop over all possible n-1 grams
    for ngram in product(alphabet, repeat=n - 1):
        idx = len(ngram2state)
        if weighted:
            p[idx] = _next_symbol_probabilities(alphabet, EOS, BOS, 10, rng)
        else:
            p[idx] = np.ones(n_symbols + 1)
        ngram2state[ngram] = idx

    A.λ = np.zeros(A.n_states)
    A.λ[ngram2state[tuple((BOS,) * (n - 1))]] = 1
    A.ρ = np.asarray([p[idx][EOS] for idx in range(A.n_states)])

    A.Ts = {}
    for y in range(n_symbols):
        A.Ts[y] = np.zeros((A.n_states, A.n_states))
        for ngram, idx in ngram2state.items():
            A.Ts[y][idx, ngram2state[tuple(ngram[1:] + (y,))]] = p[idx][y]

    A.q2str = {v: f"({','.join([str(i) for i in k])})" for k, v in ngram2state.items()}

    A.q_struct = {v: k for k, v in ngram2state.items()}

    return A


def geometric_sum_pfsa(q: float) -> PFSA:
    A = PFSA(1, 1)
    A.λ = np.array([1])
    A.ρ = np.array([1 - q])
    A.Ts = {0: np.array([[q]])}
    return A
