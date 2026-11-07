"""
Problem generators: QUBO, MaxCut, 0-1 Knapsack.

Each generator returns (Q, offset, metadata) where:
  Q : np.ndarray of shape (n, n) — symmetric QUBO matrix (upper triangular convention)
  offset : float — constant offset
  metadata : dict — instance info

For QUBO problems we follow: min x^T Q x, x ∈ {0,1}^n
All problems are cast as QUBO minimization.
"""

import numpy as np
import itertools
import os

SEED = 42

def set_seed(s: int = 42):
    np.random.seed(s)


def random_qubo(n: int, density: float = 0.5, rng=None) -> tuple:
    """Random QUBO instance with specified density."""
    if rng is None:
        rng = np.random.RandomState(SEED)
    Q = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if rng.random() < density:
                Q[i, j] = rng.uniform(-1, 1)
    # Make symmetric (Q_ij + Q_ji)/2 for i<j
    for i in range(n):
        for j in range(i+1, n):
            val = (Q[i, j] + Q[j, i]) / 2.0
            Q[i, j] = val
            Q[j, i] = val
    # Clear lower triangle for diag, keep upper for QUBO convention
    # QUBO: min sum_i Q_ii x_i + sum_{i<j} Q_ij x_i x_j
    offset = 0.0
    meta = {"type": "random_qubo", "n": n, "density": density}
    return Q, offset, meta


def maxcut_to_qubo(W: np.ndarray) -> tuple:
    """Convert MaxCut graph (weighted adjacency) to QUBO.
    MaxCut: max sum_{i<j} W_ij (1 - s_i s_j)/2, s ∈ {±1}^n
    QUBO form: min x^T Q x, x ∈ {0,1}^n
    Transformation: s = 2x - 1
    """
    n = W.shape[0]
    Q = np.zeros((n, n))
    sum_weights = np.sum(W) / 2.0
    for i in range(n):
        for j in range(i+1, n):
            if W[i, j] != 0:
                Q[i, i] += -W[i, j] / 2.0
                Q[j, j] += -W[i, j] / 2.0
                Q[i, j] += W[i, j] / 2.0
    offset = -sum_weights / 2.0
    return Q, offset, {"type": "maxcut", "n": n}


def random_maxcut(n: int, edge_prob: float = 0.5, rng=None) -> tuple:
    """Random MaxCut instance."""
    if rng is None:
        rng = np.random.RandomState(SEED)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < edge_prob:
                w = rng.uniform(0.5, 1.5)
                W[i, j] = w
                W[j, i] = w
    return maxcut_to_qubo(W)


def knapsack_to_qubo(values: np.ndarray, weights: np.ndarray, capacity: float,
                     penalty: float = None) -> tuple:
    """Convert 0-1 knapsack to QUBO using penalty method.
    max sum v_i x_i  s.t. sum w_i x_i ≤ C, x_i ∈ {0,1}
    QUBO: min -sum v_i x_i + P * (sum w_i x_i - C)^2
    """
    n = len(values)
    if penalty is None:
        penalty = np.sum(values)  # standard penalty
    # Expand (sum w_i x_i - C)^2 = (sum w_i x_i)^2 - 2C sum w_i x_i + C^2
    # = sum_i w_i^2 x_i + sum_{i≠j} w_i w_j x_i x_j - 2C sum w_i x_i + C^2
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] = -values[i] + penalty * (weights[i]**2 - 2*capacity*weights[i])
        for j in range(i+1, n):
            Q[i, j] = penalty * 2 * weights[i] * weights[j]
    offset = penalty * capacity**2
    meta = {"type": "knapsack", "n": n, "capacity": float(capacity),
            "max_value": float(np.sum(values))}
    return Q, offset, meta


def random_knapsack(n: int, rng=None) -> tuple:
    """Random 0-1 knapsack instance. Values and weights ~ U[1, 100]."""
    if rng is None:
        rng = np.random.RandomState(SEED)
    values = rng.uniform(1, 100, size=n).astype(np.float64)
    weights = rng.uniform(1, 100, size=n).astype(np.float64)
    capacity = 0.5 * np.sum(weights)
    return knapsack_to_qubo(values, weights, capacity)


def evaluate_qubo(x: np.ndarray, Q: np.ndarray, offset: float = 0.0) -> float:
    """Evaluate QUBO objective: x^T Q x + offset, x ∈ {0,1}^n."""
    return x @ Q @ x + offset


def brute_force_qubo(Q: np.ndarray, offset: float = 0.0) -> tuple:
    """Brute-force exact optimum for small n (≤25)."""
    n = Q.shape[0]
    best_val = float('inf')
    best_x = None
    for bits in itertools.product([0, 1], repeat=n):
        x = np.array(bits, dtype=float)
        val = evaluate_qubo(x, Q, offset)
        if val < best_val:
            best_val = val
            best_x = x.copy()
    return best_x, best_val


def reference_solution(Q: np.ndarray, offset: float, method: str = 'bruteforce',
                      time_limit: float = 10.0) -> dict:
    """Get reference (near-)optimal solution."""
    n = Q.shape[0]
    if n <= 20 and method == 'bruteforce':
        x, val = brute_force_qubo(Q, offset)
        return {"x": x.tolist(), "value": float(val), "method": "brute_force",
                "optimal": True}
    else:
        # Use CP-SAT for larger n (converted through binary variables)
        # For QUBO, we can use OpenJij SA as reference with many trials
        import openjij as oj
        sampler = oj.SASampler(num_reads=100, num_sweeps=1000)
        response = sampler.sample_qubo(Q)
        x = np.array(response.states[0])
        val = evaluate_qubo(x, Q, offset)
        return {"x": x.tolist(), "value": float(val),
                "method": "openjij_SA_100runs", "optimal": False}


def generate_benchmark(sizes=(20, 30, 50), n_instances=10, 
                       problems=('random_qubo', 'maxcut', 'knapsack'),
                       output_dir='results/raw'):
    """Generate all benchmark instances and save to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Use fixed seed for reproducibility
    rng = np.random.RandomState(SEED + 42)
    
    index = []
    for pb in problems:
        for n in sizes:
            for inst in range(n_instances):
                seed = rng.randint(0, 10000)
                rng_instance = np.random.RandomState(seed)
                
                if pb == 'random_qubo':
                    Q, off, meta = random_qubo(n, density=0.3, rng=rng_instance)
                elif pb == 'maxcut':
                    Q, off, meta = random_maxcut(n, edge_prob=0.5, rng=rng_instance)
                elif pb == 'knapsack':
                    Q, off, meta = random_knapsack(n, rng=rng_instance)
                
                # Save QUBO matrix
                fname = f"{pb}_n{n}_inst{inst:02d}"
                np.savez(os.path.join(output_dir, f"{fname}.npz"),
                         Q=Q, offset=np.float64(off))
                
                index.append({"name": fname, "problem": pb, "n": n,
                             "instance": inst, "seed": seed})
    
    import pandas as pd
    df = pd.DataFrame(index)
    df.to_csv(os.path.join(output_dir, "benchmark_index.csv"), index=False)
    print(f"Generated {len(index)} instances -> {output_dir}")
    return df


if __name__ == '__main__':
    df = generate_benchmark()
    print(df.groupby(['problem', 'n']).size().to_string())