from nltk.util import pad_sequence, ngrams
from itertools import chain
from functools import partial

from functools import partial
from itertools import chain

from nltk.util import everygrams, pad_sequence


flatten = chain.from_iterable


def pad_eos(text):
    return pad_sequence(
        text,
        pad_left=False,
        pad_right=True,
        right_pad_symbol="</s>",
        n=2,
    )


def pad_bos(text, n):
    return pad_sequence(
        text,
        pad_left=True,
        left_pad_symbol="<s>",
        pad_right=False,
        n=n,
    )


def pad_both_ends(text, n):
    eos_padded = pad_eos(text)
    both_padded = pad_bos(eos_padded, n)
    return both_padded


def eos_ngram_pipeline(n, text, add_bos):
    if add_bos:
        pad_bos = partial(
            pad_sequence,
            pad_left=True,
            pad_right=False,
            left_pad_symbol="<s>",
        )
        padded_text = (list(pad_bos(sent, n)) + ["</s>"] for sent in text)
        print(list(pad_bos(text[0], n)) + ["</s>"])
        print(list(ngrams(pad_bos(text[0], n), n)))
        print(list(pad_bos(text[1], n)) + ["</s>"])
        print(list(ngrams(pad_bos(text[1], n), n)))
        print(list(pad_bos(text[2], n)) + ["</s>"])
        print(list(ngrams(pad_bos(text[2], n), n)))
    else:
        padded_text = (sent + ["</s>"] for sent in text)

    # Generate n-grams (using nltk.util.ngrams)
    train_data = (list(ngrams(sent, n)) for sent in padded_text)
    vocab = flatten(padded_text)

    return train_data, vocab


def padded_everygram_pipeline(order, text, add_bos):
    if add_bos:
        padding_fn = partial(pad_both_ends, n=order)
    else:
        padding_fn = pad_eos

    return (
        (everygrams(list(padding_fn(sent)), max_len=order) for sent in text),
        flatten(map(padding_fn, text)),
    )
