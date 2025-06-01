import numpy as np


class CharRNN:

    # initialiser                                                        
    def __init__(
        self,
        K: int,
        m: int = 100,
        eta: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        seed: int = 42,
    ) -> None:
        self.K, self.m = K, m
        self.eta, self.beta1, self.beta2, self.eps = eta, beta1, beta2, eps

        rng = np.random.default_rng(seed)
        s2 = np.sqrt(2)
        self.params = dict(
            U=rng.standard_normal((m, K)) / (s2 * np.sqrt(K)),
            W=rng.standard_normal((m, m)) / (s2 * np.sqrt(m)),
            V=rng.standard_normal((K, m)) / np.sqrt(m),
            b=np.zeros((m, 1)),
            c=np.zeros((K, 1)),
        )

        # Adam buffers
        self.mom1 = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.mom2 = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0

    # forward                                                            
    def _forward(self, X: np.ndarray, Y: np.ndarray, hprev: np.ndarray):
        U, W, V, b, c = (self.params[k] for k in ("U", "W", "V", "b", "c"))
        τ = X.shape[1]

        H = np.zeros((self.m, τ))
        A = np.zeros_like(H)
        P = np.zeros((self.K, τ))

        h_t = hprev
        for t in range(τ):
            x_t = X[:, t:t + 1]
            a_t = W @ h_t + U @ x_t + b
            h_t = np.tanh(a_t)
            o_t = V @ h_t + c
            p_t = np.exp(o_t) / np.sum(np.exp(o_t), keepdims=True, axis=0)

            A[:, t:t + 1] = a_t
            H[:, t:t + 1] = h_t
            P[:, t:t + 1] = p_t

        # loss averaged over τ steps 
        loss = -np.sum(np.log(np.sum(P * Y, axis=0) + 1e-12)) / τ

        cache = dict(X=X, Y=Y, H=H, A=A, P=P, hprev=hprev, τ=τ)
        return loss, cache, h_t

    # backward                                                           #
    def _backward(self, cache):
        U, W, V, b, c = (self.params[k] for k in ("U", "W", "V", "b", "c"))
        X, Y, H, A, P, τ = (cache[k] for k in ("X", "Y", "H", "A", "P", "τ"))

        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        dh_next = np.zeros((self.m, 1))

        for t in reversed(range(τ)):
            x_t = X[:, t:t + 1]
            y_t = Y[:, t:t + 1]
            p_t = P[:, t:t + 1]

            do = p_t - y_t
            grads["V"] += do @ H[:, t:t + 1].T
            grads["c"] += do

            dh = V.T @ do + dh_next
            da = dh * (1 - np.tanh(A[:, t:t + 1]) ** 2)
            grads["b"] += da
            grads["U"] += da @ x_t.T
            h_prev_t = H[:, t - 1:t] if t else cache["hprev"]
            grads["W"] += da @ h_prev_t.T

            dh_next = W.T @ da


        for k in grads:
            grads[k] /= τ

        return grads


    # Adam update (with clipping)                                        
    def _adam_step(self, grads):
        self.t += 1
        for k, g in grads.items():
            np.clip(g, -5, 5, out=g)  # protects training, not used in check

            self.mom1[k] = self.beta1 * self.mom1[k] + (1 - self.beta1) * g
            self.mom2[k] = self.beta2 * self.mom2[k] + (1 - self.beta2) * g**2

            m_hat = self.mom1[k] / (1 - self.beta1 ** self.t)
            v_hat = self.mom2[k] / (1 - self.beta2 ** self.t)

            self.params[k] -= self.eta * m_hat / (np.sqrt(v_hat) + self.eps)


    # public train-step                                                  #
    def train_step(self, X: np.ndarray, Y: np.ndarray, hprev: np.ndarray):
        loss, cache, h_last = self._forward(X, Y, hprev)
        grads = self._backward(cache)
        self._adam_step(grads)
        return loss, h_last

    # sampler                                                            

    def sample(self, h0, x0, n, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        U, W, V, b, c = (self.params[k] for k in ("U", "W", "V", "b", "c"))

        h = h0.copy()
        x = x0.copy()
        out = np.zeros((self.K, n))
        for t in range(n):
            h = np.tanh(W @ h + U @ x + b)
            p = np.exp(V @ h + c)
            p /= p.sum()
            idx = np.searchsorted(np.cumsum(p).ravel(), rng.uniform())
            x = np.zeros_like(x)
            x[idx, 0] = 1
            out[:, t:t + 1] = x
        return out
