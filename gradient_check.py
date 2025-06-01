
import numpy as np
import torch
import char_rnn, data_utils
from torch_gradient_computations_column_wise import ComputeGradsWithTorch


def main():
    txt = "This is only a test string for tiny gradient checking."
    _, K, c2i, _ = data_utils.build_vocab(txt)

    rnn = char_rnn.CharRNN(K, m=10, eta=1e-3, seed=0)

    seq_len = 25
    rng = np.random.default_rng(0)
    s = rng.integers(0, len(txt) - seq_len - 1)
    xchars = txt[s : s + seq_len]
    ychars = txt[s + 1 : s + seq_len + 1]

    X = data_utils.chars_to_onehot(xchars, c2i)
    Y = data_utils.chars_to_onehot(ychars, c2i)
    h0 = np.zeros((10, 1))

    # analytic
    _, cache, _ = rnn._forward(X, Y, h0)
    grads = rnn._backward(cache)

    # autograd
    y_idx = Y.argmax(axis=0)
    torch_grads = ComputeGradsWithTorch(X, y_idx, h0, rnn.params)

    for k in grads:
        err = np.abs(grads[k] - torch_grads[k]).max()
        print(f"{k:>2}: max d = {err:.3e}")
        assert err < 1e-6


if __name__ == "__main__":
    main()
