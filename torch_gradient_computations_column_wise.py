# torch_gradient_computations_column_wise.py

import torch
import numpy as np

def ComputeGradsWithTorch(X, y, h0, RNN):
    """
    COLUMN-WISE PyTorch gradient computation.

    X:    NumPy array of shape (d x tau)   – each COLUMN is a one-hot input vector.
    y:    length-tau list/array of target indices.
    h0:   NumPy array of shape (m x 1) – initial hidden state (column vector).
    RNN:  dict with keys 'U','W','V','b','c' (all NumPy arrays).

    Returns: dict grads with same keys, each a NumPy array of gradients.
    """
    tau = X.shape[1]       # number of timesteps
    d   = X.shape[0]       # input dimension (== vocabulary size K)
    m   = h0.shape[0]      # hidden size
    K   = RNN['V'].shape[0]  # output dimension (vocabulary size)

    # Convert inputs to torch tensors
    Xt = torch.from_numpy(X).double()   # shape: (d x tau)
    ht = torch.from_numpy(h0).double()  # shape: (m x 1)

    torch_network = {}
    for kk in RNN.keys():
        torch_network[kk] = torch.tensor(RNN[kk], dtype=torch.float64, requires_grad=True)

    apply_tanh    = torch.nn.Tanh()
    apply_softmax = torch.nn.Softmax(dim=0)  # softmax down each column

    # Allocate space to store hidden states: (m x tau)
    Hs = torch.empty((m, tau), dtype=torch.float64)

    hprev = ht  # (m x 1)
    # -------- forward recursion (column-wise) --------
    for t in range(tau):
        x_t = Xt[:, t:t+1]   # (d x 1)

        # a_t = W @ hprev + U @ x_t + b
        a_t = torch_network['W'].mm(hprev) \
              + torch_network['U'].mm(x_t) \
              + torch_network['b']       # (m x 1)
        h_t = apply_tanh(a_t)           # (m x 1)
        Hs[:, t:t+1] = h_t              # store hidden as column t
        hprev = h_t                     # for next timestep

    # -------- compute outputs and loss --------
    #   V: (K x m), Hs: (m x tau) → V.mm(Hs) = (K x tau)
    Os = torch_network['V'].mm(Hs) + torch_network['c']  # (K x tau), c: (K x 1) broadcast
    P  = apply_softmax(Os)                               # (K x tau)

    # Cross-entropy over each column
    y_tensor = torch.tensor(y, dtype=torch.long)         # (tau,)
    Pt = P[y_tensor, torch.arange(tau)]                  # shape (tau,)
    loss = torch.mean(-torch.log(Pt))                    # scalar

    # -------- backward pass --------
    loss.backward()

    grads = {}
    for kk in RNN.keys():
        grads[kk] = torch_network[kk].grad.numpy()

    return grads
