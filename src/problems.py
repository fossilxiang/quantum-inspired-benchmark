
"""Problem generators: Random QUBO, MaxCut, 0-1 Knapsack."""
import numpy as np, itertools, os

def random_qubo(n, density=0.3, seed=42):
    rng = np.random.RandomState(seed)
    Q = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if rng.random() < density:
                Q[i, j] = rng.uniform(-1, 1)
    for i in range(n):
        for j in range(i+1, n):
            v = (Q[i,j] + Q[j,i]) / 2.0
            Q[i,j] = Q[j,i] = v
    return Q, 0.0, {"type": "random_qubo", "n": n, "density": density}

def random_maxcut(n, edge_prob=0.5, seed=42):
    rng = np.random.RandomState(seed)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < edge_prob:
                w = rng.uniform(0.5, 1.5)
                W[i,j] = W[j,i] = w
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i,i] = -np.sum(W[i]) / 2.0
        for j in range(i+1, n):
            if W[i,j] != 0:
                Q[i,j] = W[i,j] / 2.0
    offset = -np.sum(W) / 4.0
    return Q, offset, {"type": "maxcut", "n": n}

def knapsack_to_qubo(values, weights, capacity, penalty=None):
    n = len(values)
    if penalty is None: penalty = np.sum(values)
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i,i] = -values[i] + penalty * (weights[i]**2 - 2*capacity*weights[i])
        for j in range(i+1, n):
            Q[i,j] = penalty * 2 * weights[i] * weights[j]
    offset = penalty * capacity**2
    return Q, offset

def random_knapsack(n, seed=42):
    rng = np.random.RandomState(seed)
    v = rng.uniform(1, 100, n)
    w = rng.uniform(1, 100, n)
    cap = 0.5 * np.sum(w)
    return knapsack_to_qubo(v, w, cap)

def brute_force_qubo(Q, offset=0.0):
    n = Q.shape[0]
    best_val = float('inf'); best_x = None
    for bits in itertools.product([0,1], repeat=n):
        x = np.array(bits, dtype=float)
        val = x @ Q @ x + offset
        if val < best_val: best_val, best_x = val, x.copy()
    return best_x, best_val
