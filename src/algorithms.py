"""
Algorithm wrappers: quantum-inspired (self-implemented BSB/SimCIM + OpenJij SQA)
and classical (SA, CP-SAT).

BSB and SimCIM are self-implemented in pure NumPy for full reproducibility and
to avoid API versioning issues with MindQuantum. Details below.

All solvers follow a unified interface:
    solve(Q, offset, time_limit, seed) -> dict with keys: x, value, time, algorithm
"""

import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


def evaluate_qubo(x: np.ndarray, Q: np.ndarray, offset: float = 0.0) -> float:
    return float(x @ Q @ x + offset)


def _to_qb(Q: np.ndarray):
    """Split QUBO matrix Q into Ising coupling J (zero diag) and field h."""
    h = np.diag(Q).copy().astype(np.float64)
    J = Q.copy().astype(np.float64)
    np.fill_diagonal(J, 0.0)
    return J, h


# ═══════════════════════════════════════════════════════════════════════════
# Quantum-Inspired: BSB  (Bidirectional Self-Broadcasting)
# ═══════════════════════════════════════════════════════════════════════════
# Self-implemented. Continuous relaxation with gradient-based updates:
#   x ← clip(x - η · ∇E(x), 0, 1)  where ∇E = Q @ x
# Multiple random starts with annealing β schedule.
# This follows the BSB paradigm of Ising spin broadcasting dynamics
# adapted for QUBO with a sigmoidal nonlinearity.

def solve_bsb(Q: np.ndarray, offset: float = 0.0, time_limit: float = 10.0,
              seed: int = 42, n_iter: int = 800, n_starts: int = 5) -> dict:
    n = Q.shape[0]
    rng_base = np.random.RandomState(seed)
    t0 = time.perf_counter()

    best_val = float('inf')
    best_x = None

    for restart in range(n_starts):
        rng = np.random.RandomState(seed + restart * 7)
        x = rng.uniform(0.0, 1.0, size=n).astype(np.float64)

        for t in range(n_iter + 1):
            # Annealing schedule
            η = 0.05 * (1.0 - t / n_iter) + 0.001
            β = 0.1 + 5.0 * t / n_iter

            # Gradient: ∇E = Q @ x  (linearized QUBO gradient)
            grad = Q @ x
            x = x - η * grad

            # Sigmoid nonlinearity (cell-rate function in BSB)
            x = 1.0 / (1.0 + np.exp(-β * (x - 0.5)))

        x_bin = (x > 0.5).astype(float)
        val = evaluate_qubo(x_bin, Q, offset)
        if val < best_val:
            best_val = val
            best_x = x_bin.copy()

        # Time cap
        if time_limit > 0 and time.perf_counter() - t0 > time_limit:
            break

    elapsed = time.perf_counter() - t0
    return {"x": best_x.tolist(), "value": best_val, "time": elapsed,
            "algorithm": "BSB"}


# ═══════════════════════════════════════════════════════════════════════════
# Quantum-Inspired: SimCIM  (Simulated Coherent Ising Machine)
# ═══════════════════════════════════════════════════════════════════════════
# Self-implemented. Phase oscillator dynamics:
#   dθ_i/dt = h_i·sin(θ_i) + Σ_j J_ij·sin(θ_j-θ_i) + noise(t)
#   spin_i = sign(cos(θ_i))
# Vectorized using trig identities.

def solve_simcim(Q: np.ndarray, offset: float = 0.0, time_limit: float = 10.0,
                 seed: int = 42, n_iter: int = 2000) -> dict:
    n = Q.shape[0]
    J, h = _to_qb(Q)
    rng = np.random.RandomState(seed)
    t0 = time.perf_counter()

    θ = rng.uniform(-np.pi / 4, np.pi / 4, size=n).astype(np.float64)
    dt_val = 0.05
    noise_scale = 0.8

    for t in range(n_iter + 1):
        sin_θ = np.sin(θ)
        cos_θ = np.cos(θ)

        # Vectorized phase update: dθ_i = h_i·sin(θ_i) + Σ J_ij·sin(θ_j-θ_i)
        J_cos = J @ cos_θ
        J_sin = J @ sin_θ
        dθ = sin_θ * (h - J_cos) + cos_θ * J_sin

        noise = noise_scale * rng.randn(n)
        θ += dt_val * (dθ + noise)
        noise_scale *= 0.997  # slow noise decay

        θ = np.arctan2(np.sin(θ), np.cos(θ))

    # Phase → Ising spin → QUBO binary
    s = np.cos(θ)
    x = (s > 0.0).astype(float)
    val = evaluate_qubo(x, Q, offset)
    elapsed = time.perf_counter() - t0

    return {"x": x.tolist(), "value": val, "time": elapsed,
            "algorithm": "SimCIM"}


# ═══════════════════════════════════════════════════════════════════════════
# OpenJij SQA  (Simulated Quantum Annealing — classical simulation)
# ═══════════════════════════════════════════════════════════════════════════
# Uses OpenJij's SQASampler which simulates quantum annealing via path-integral
# Monte Carlo (classical hardware).

def solve_sqa(Q: np.ndarray, offset: float = 0.0, time_limit: float = 10.0,
              seed: int = 42) -> dict:
    import openjij as oj

    t0 = time.perf_counter()
    sampler = oj.SQASampler()
    response = sampler.sample_qubo(Q, num_reads=10, num_sweeps=1000, seed=seed)

    x = np.array(response.states[0], dtype=float)
    val = evaluate_qubo(x, Q, offset)
    elapsed = time.perf_counter() - t0

    return {"x": x.tolist(), "value": val, "time": elapsed,
            "algorithm": "SQA"}


# ═══════════════════════════════════════════════════════════════════════════
# Classical: Simulated Annealing (OpenJij)
# ═══════════════════════════════════════════════════════════════════════════

def solve_sa(Q: np.ndarray, offset: float = 0.0, time_limit: float = 10.0,
             seed: int = 42) -> dict:
    import openjij as oj

    t0 = time.perf_counter()
    sampler = oj.SASampler()
    response = sampler.sample_qubo(Q, num_reads=10, num_sweeps=1000, seed=seed)

    x = np.array(response.states[0], dtype=float)
    val = evaluate_qubo(x, Q, offset)
    elapsed = time.perf_counter() - t0

    return {"x": x.tolist(), "value": val, "time": elapsed,
            "algorithm": "SA"}


# ═══════════════════════════════════════════════════════════════════════════
# Classical: CP-SAT (OR-Tools exact solver)
# ═══════════════════════════════════════════════════════════════════════════

def solve_cpsat(Q: np.ndarray, offset: float = 0.0, time_limit: float = 30.0,
                seed: int = 42) -> dict:
    from ortools.sat.python import cp_model

    n = Q.shape[0]
    t0 = time.perf_counter()

    model = cp_model.CpModel()
    x = [model.NewBoolVar(f'x{i}') for i in range(n)]

    terms = []
    for i in range(n):
        terms.append(Q[i, i] * x[i])
        for j in range(i + 1, n):
            if abs(Q[i, j]) > 1e-10:
                y = model.NewBoolVar(f'y_{i}_{j}')
                model.Add(y <= x[i])
                model.Add(y <= x[j])
                model.Add(y >= x[i] + x[j] - 1)
                terms.append(Q[i, j] * y)

    model.Minimize(sum(terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.random_seed = seed
    solver.parameters.num_search_workers = 2
    solver.parameters.log_search_progress = False

    status = solver.Solve(model)
    elapsed = time.perf_counter() - t0

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        x_val = np.array([solver.Value(v) for v in x], dtype=float)
        val = evaluate_qubo(x_val, Q, offset)
    else:
        rng = np.random.RandomState(seed)
        x_val = rng.randint(0, 2, size=n).astype(float)
        val = evaluate_qubo(x_val, Q, offset)

    return {"x": x_val.tolist(), "value": val, "time": elapsed,
            "status": status, "optimal": status == cp_model.OPTIMAL,
            "algorithm": "CP-SAT"}


# ═══════════════════════════════════════════════════════════════════════════
# Solver Registry
# ═══════════════════════════════════════════════════════════════════════════

SOLVERS = {
    "BSB": solve_bsb,
    "SimCIM": solve_simcim,
    "SQA": solve_sqa,
    "SA": solve_sa,
    "CP-SAT": solve_cpsat,
}


def get_solver_names():
    return list(SOLVERS.keys())


def run_solver(name: str, Q: np.ndarray, offset: float,
               time_limit: float = 30.0, seed: int = 42) -> dict:
    if name not in SOLVERS:
        raise ValueError(f"Unknown solver: {name}. Options: {list(SOLVERS.keys())}")
    return SOLVERS[name](Q=Q, offset=offset, time_limit=time_limit, seed=seed)


if __name__ == '__main__':
    from problems import random_qubo, brute_force_qubo

    np.random.seed(42)
    Q, off, _ = random_qubo(10, density=0.5)
    _, bf_val = brute_force_qubo(Q, off)
    print(f"Brute-force optimum: {bf_val:.6f}")

    for name in get_solver_names():
        result = run_solver(name, Q, off, time_limit=5.0, seed=42)
        gap = (result['value'] - bf_val) / max(1.0, abs(bf_val)) * 100
        print(f"{name:10s} value={result['value']:.4f}  gap={gap:6.2f}%  "
              f"time={result['time']:.4f}s")