
"""Self-implemented quantum-inspired + classical solvers."""
import time, numpy as np, warnings
warnings.filterwarnings('ignore')

def evaluate_qubo(x, Q, offset=0.0): return float(x @ Q @ x + offset)

def _to_qb(Q):
    h = np.diag(Q).copy().astype(np.float64)
    J = Q.copy().astype(np.float64)
    np.fill_diagonal(J, 0.0)
    return J, h

def solve_bsb(Q, offset=0.0, time_limit=10.0, seed=42, n_iter=800, n_starts=5):
    """Gradient-based QIR: continuous relaxation + sigmoid."""
    n = Q.shape[0]; t0 = time.perf_counter()
    best_val = float('inf'); best_x = None
    for restart in range(n_starts):
        rng = np.random.RandomState(seed + restart * 7)
        x = rng.uniform(0, 1, size=n).astype(np.float64)
        for t in range(n_iter + 1):
            eta = 0.05 * (1 - t/n_iter) + 0.001
            beta = 0.1 + 5.0 * t / n_iter
            grad = Q @ x
            x = x - eta * grad
            x = 1.0 / (1.0 + np.exp(-beta * (x - 0.5)))
        x_bin = (x > 0.5).astype(float)
        val = evaluate_qubo(x_bin, Q, offset)
        if val < best_val: best_val, best_x = val, x_bin.copy()
        if time_limit > 0 and time.perf_counter() - t0 > time_limit: break
    return {"x": best_x.tolist(), "value": best_val, "time": time.perf_counter() - t0, "algorithm": "GQIR"}

def solve_simcim(Q, offset=0.0, time_limit=10.0, seed=42, n_iter=2000):
    """Self-implemented SimCIM phase dynamics."""
    n = Q.shape[0]; J, h = _to_qb(Q)
    rng = np.random.RandomState(seed); t0 = time.perf_counter()
    theta = rng.uniform(-np.pi/4, np.pi/4, n).astype(np.float64)
    dt_val, noise_scale = 0.05, 0.8
    for t in range(n_iter + 1):
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        J_cos, J_sin = J @ cos_t, J @ sin_t
        dtheta = sin_t * (h - J_cos) + cos_t * J_sin
        theta += dt_val * (dtheta + noise_scale * rng.randn(n))
        noise_scale *= 0.997
        theta = np.arctan2(np.sin(theta), np.cos(theta))
    s = np.cos(theta); x = (s > 0.0).astype(float)
    val = evaluate_qubo(x, Q, offset)
    return {"x": x.tolist(), "value": val, "time": time.perf_counter() - t0, "algorithm": "SimCIM"}

def solve_sqa(Q, offset=0.0, time_limit=10.0, seed=42):
    import openjij as oj
    t0 = time.perf_counter()
    sampler = oj.SQASampler()
    response = sampler.sample_qubo(Q, num_reads=10, num_sweeps=1000, seed=seed)
    x = np.array(response.states[0], dtype=float)
    val = evaluate_qubo(x, Q, offset)
    return {"x": x.tolist(), "value": val, "time": time.perf_counter() - t0, "algorithm": "SQA"}

def solve_sa(Q, offset=0.0, time_limit=10.0, seed=42):
    import openjij as oj
    t0 = time.perf_counter()
    sampler = oj.SASampler()
    response = sampler.sample_qubo(Q, num_reads=10, num_sweeps=1000, seed=seed)
    x = np.array(response.states[0], dtype=float)
    val = evaluate_qubo(x, Q, offset)
    return {"x": x.tolist(), "value": val, "time": time.perf_counter() - t0, "algorithm": "SA"}

def solve_cpsat(Q, offset=0.0, time_limit=30.0, seed=42):
    from ortools.sat.python import cp_model
    n = Q.shape[0]; t0 = time.perf_counter()
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x{i}") for i in range(n)]
    terms = []
    for i in range(n):
        terms.append(Q[i, i] * x[i])
        for j in range(i+1, n):
            if abs(Q[i, j]) > 1e-10:
                y = model.NewBoolVar(f"y_{i}_{j}")
                model.Add(y <= x[i]); model.Add(y <= x[j])
                model.Add(y >= x[i] + x[j] - 1)
                terms.append(Q[i, j] * y)
    model.Minimize(sum(terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.random_seed = seed
    solver.parameters.num_search_workers = 2
    status = solver.Solve(model)
    elapsed = time.perf_counter() - t0
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        x_val = np.array([solver.Value(v) for v in x], dtype=float)
        val = evaluate_qubo(x_val, Q, offset)
    else:
        rng = np.random.RandomState(seed)
        x_val = rng.randint(0, 2, size=n).astype(float)
        val = evaluate_qubo(x_val, Q, offset)
    return {"x": x_val.tolist(), "value": val, "time": elapsed, "status": status, "algorithm": "CP-SAT"}

SOLVERS = {"BSB": solve_bsb, "SimCIM": solve_simcim, "SQA": solve_sqa, "SA": solve_sa, "CP-SAT": solve_cpsat}
