import random

MARKER_REVERSE = "REVERSE"


class PerturbationFunc:
    def __init__(self, **kwargs):
        super().__init__()

    def perturb(self, sentence: str, **kwargs) -> str:
        raise NotImplementedError


class NoReverse(PerturbationFunc):
    def perturb(self, sent):
        tokens = sent.split(" ")

        # Remove and store [eos]
        eos = tokens.pop() if tokens[-1] == "[eos]" else None

        # Insert REVERSE marker at random position
        insert_pos = random.randint(0, len(tokens))
        tokens.insert(insert_pos, MARKER_REVERSE)

        if eos:
            tokens.append(eos)
        return " ".join(tokens)


class PartialReverse(PerturbationFunc):
    def perturb(self, sent):
        tokens = sent.split(" ")

        # Remove and store [eos]
        eos = tokens.pop() if tokens[-1] == "[eos]" else None

        # Insert REVERSE marker and reverse tokens after it
        insert_pos = random.randint(0, len(tokens))
        tokens.insert(insert_pos, MARKER_REVERSE)

        # Reverse tokens after the marker
        tokens[insert_pos + 1 :] = tokens[insert_pos + 1 :][::-1]

        if eos:
            tokens.append(eos)
        return " ".join(tokens)


class FullReverse(PerturbationFunc):
    def perturb(self, sent):
        tokens = sent.split(" ")

        # Remove and store [eos]
        eos = tokens.pop() if tokens[-1] == "[eos]" else None

        # Insert REVERSE marker at random position
        insert_pos = random.randint(0, len(tokens))
        tokens.insert(insert_pos, MARKER_REVERSE)

        # Reverse all tokens
        tokens = tokens[::-1]

        if eos:
            tokens.append(eos)
        return " ".join(tokens)


class Reverse(PerturbationFunc):
    # this is not in Kallini's
    def perturb(self, sent):
        tokens = sent.split(" ")

        # Remove and store [eos]
        eos = tokens.pop() if tokens[-1] == "[eos]" else None
        # Reverse all tokens
        tokens = tokens[::-1]

        if eos:
            tokens.append(eos)
        return " ".join(tokens)


class DeterministicShuffle(PerturbationFunc):
    def __init__(self, seed):
        super().__init__()
        self.seed = seed

    def perturb(self, sent):
        # Save current random state
        state = random.getstate()

        tokens = sent.split(" ")
        eos = tokens.pop() if tokens[-1] == "[eos]" else None

        # Use seed for shuffling
        random.seed(self.seed)
        random.shuffle(tokens)

        # Restore random state after shuffling
        random.setstate(state)

        if eos:
            tokens.append(eos)
        return " ".join(tokens)


class NonDeterministicShuffle(PerturbationFunc):
    def perturb(self, sent):
        tokens = sent.split(" ")
        eos = tokens.pop() if tokens[-1] == "[eos]" else None

        random.shuffle(tokens)

        if eos:
            tokens.append(eos)
        return " ".join(tokens)


class LocalShuffle(PerturbationFunc):
    def __init__(self, seed, window):
        super().__init__()
        self.seed = seed
        self.window = window

    def perturb(self, sent):
        tokens = sent.split(" ")
        eos = tokens.pop() if tokens[-1] == "[eos]" else None

        # Save current random state
        state = random.getstate()

        # Use seed for shuffling
        random.seed(self.seed)
        shuffled_tokens = []
        for i in range(0, len(tokens), self.window):
            batch = tokens[i : min(i + self.window, len(tokens))].copy()
            random.shuffle(batch)
            shuffled_tokens.extend(batch)

        # Restore random state after shuffling
        random.setstate(state)

        if eos:
            shuffled_tokens.append(eos)
        return " ".join(shuffled_tokens)


class EvenOddShuffle(PerturbationFunc):
    def perturb(self, sent):
        tokens = sent.split(" ")
        eos = tokens.pop() if tokens[-1] == "[eos]" else None

        even = [tok for i, tok in enumerate(tokens) if i % 2 == 0]
        odd = [tok for i, tok in enumerate(tokens) if i % 2 != 0]
        shuffled = even + odd

        if eos:
            shuffled.append(eos)
        return " ".join(shuffled)


class OddEvenShuffle(PerturbationFunc):
    def perturb(self, sent):
        tokens = sent.split(" ")
        eos = tokens.pop() if tokens[-1] == "[eos]" else None

        odd = [tok for i, tok in enumerate(tokens) if i % 2 != 0]
        even = [tok for i, tok in enumerate(tokens) if i % 2 == 0]
        shuffled = odd + even

        if eos:
            shuffled.append(eos)
        return " ".join(shuffled)
