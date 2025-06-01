import numpy as np
from pathlib import Path


def read_book(path: str | Path) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def build_vocab(book_str: str):
    chars = sorted(list(set(book_str)))
    K = len(chars)
    char_to_ind = {ch: i for i, ch in enumerate(chars)}
    ind_to_char = {i: ch for ch, i in char_to_ind.items()}
    return chars, K, char_to_ind, ind_to_char


def one_hot(indices: np.ndarray, K: int) -> np.ndarray:
    X = np.zeros((K, indices.size))
    X[indices, np.arange(indices.size)] = 1.0
    return X


def chars_to_onehot(seq: str, char_to_ind: dict) -> np.ndarray:
    idx = np.fromiter((char_to_ind[ch] for ch in seq), dtype=np.int64)
    return one_hot(idx, K=len(char_to_ind))


def onehot_to_string(X: np.ndarray, ind_to_char: dict) -> str:
    inds = X.argmax(axis=0)
    return ''.join(ind_to_char[i] for i in inds)
