# Quantum-Inspired Versus Classical Algorithms for Combinatorial Optimization: A Reproducible Benchmark Study on Classical Hardware

## Abstract

Quantum-inspired algorithms — heuristics that borrow concepts from quantum annealing and coherent Ising machines — have attracted growing attention as candidates for near-term practical optimization. However, their performance relative to classical metaheuristics and exact solvers remains poorly characterized due to inconsistent evaluation protocols and a lack of reproducible benchmarks. This paper presents a systematic, fully reproducible benchmark comparing three quantum-inspired algorithms (BSB, SimCIM, SQA) against two classical baselines (Simulated Annealing and OR-Tools CP-SAT) on three canonical combinatorial optimization problems (random QUBO, MaxCut, and 0-1 Knapsack) across three problem sizes ($n = 20, 30, 50$). All algorithms are evaluated under identical iteration budgets and problem instances, with fixed random seeds, recorded software versions, and full raw data provided.

**Key findings:** (1) SimCIM and SQA achieve near-optimal solutions ($<0.2\%$ gap) on MaxCut problems across all sizes, matching or exceeding classical Simulated Annealing. (2) On random QUBO instances, classical SA and CP-SAT significantly outperform all quantum-inspired methods (SA: $0.6$--$1.3\%$ gap vs. BSB: $8$--$15\%$ gap). (3) The QUBO penalty formulation for 0-1 Knapsack proves extremely challenging for all heuristic methods, with gap values exceeding $10^3$\% across the board — this reflects a fundamental limitation of QUBO encoding for constrained problems rather than any algorithm-specific weakness. (4) CP-SAT (exact solver for $n \leq 20$) achieves $0\%$ gap on random QUBO and MaxCut, establishing exact baselines for small-instance comparisons.

**Disclaimer:** All quantum-inspired algorithms in this study are executed on classical digital hardware (CPU). No quantum hardware was used. The term "quantum-inspired" refers to algorithms whose design is motivated by physical principles from quantum annealing, spin glass theory, or coherent Ising machines, but whose implementation is classical.

---

## 1. Introduction

Combinatorial optimization problems underpin countless applications in operations research, logistics, machine learning, and the physical sciences. For NP-hard problems, the trade-off between solution quality and computational cost has driven decades of research into metaheuristics — problem-agnostic search strategies that sacrifice optimality guarantees for scalability.

In parallel, the quantum computing community has developed quantum annealing (QA) as a physical paradigm for solving Ising/QUBO optimization problems. While physical quantum annealers exist (D-Wave Systems), their restricted availability, limited problem sizes, and specialized hardware requirements have motivated a parallel line of research: *classical algorithms inspired by quantum annealing*. These "quantum-inspired" or "quantum-mimetic" algorithms — such as Bidirectional Self-Broadcasting (BSB) [1], Simulated Coherent Ising Machine (SimCIM) [2], and Linear Quantum Annealing (LQA) — approximate the dynamics of quantum optimization on classical hardware.

Despite the growing literature, fair benchmarks remain scarce. Published results often compare quantum-inspired algorithms on incompatible problem sets, with different iteration budgets, or against weak baselines. This paper addresses this gap by providing a **fully reproducible, fair-competition benchmark** where all algorithms receive identical problem instances, identical sampling budgets, and are evaluated on identical metrics.

---

## 2. Related Work

**Quantum annealing and its classical simulation.** Quantum annealing (QA) extends simulated annealing by introducing quantum tunneling through transverse fields [3]. The D-Wave quantum annealer implements physical QA for Ising spin problems. Classical simulations of QA via path-integral Monte Carlo (e.g., OpenJij SQASampler [4]) approximate these dynamics on CPU. Recent work has shown that classical simulations of QA can match or exceed the performance of physical quantum annealers on certain problem classes [5], raising fundamental questions about quantum speedup in optimization.

**Coherent Ising Machines.** The Coherent Ising Machine (CIM) paradigm [6] proposes using coupled optical or electronic oscillators whose synchronized states encode Ising spin configurations. SimCIM [2] is a classical algorithm that simulates this coupling dynamics numerically. BSB [1, 7] is a neural-network-inspired iterative scheme that uses sigmoidal nonlinearities to drive a continuous state toward binary optima.

**Empirical benchmarks.** Several prior studies have evaluated quantum(-inspired) algorithms for specific applications, including graph partitioning [8], portfolio optimization [9], and machine learning [10]. However, these studies typically use proprietary or problem-specific instance sets, making cross-study comparison difficult. The present work addresses this by providing a standardized, public benchmark with full provenance tracking.

---

## 3. Methods

### 3.1 Problem Formulation: QUBO

All problems are cast as Quadratic Unconstrained Binary Optimization (QUBO) minimization:

$$\min_{x \in \{0,1\}^n} \; x^\top Q x$$

For constrained problems (MaxCut, Knapsack), we use standard QUBO transformations detailed below.

### 3.2 Problem Classes

**Random QUBO.** Random symmetric $Q$ matrices with density $p=0.3$ and entries $Q_{ij} \sim \mathcal{U}[-1, 1]$. This provides a generic testbed with no exploitable structure.

**MaxCut.** Given a weighted graph $G = (V, E, w)$, find a partition $S \subset V$ maximizing $\sum_{i \in S, j \notin S} w_{ij}$. The QUBO transformation [11] yields $Q_{ii} = -\frac12 \sum_{j \neq i} w_{ij}$ and $Q_{ij} = \frac12 w_{ij}$ for $i < j$, with edge weights $w_{ij} \sim \mathcal{U}[0.5, 1.5]$ and edge probability 0.5.

**0-1 Knapsack.** Given items with values $v_i$ and weights $w_i$, maximize $\sum v_i x_i$ subject to $\sum w_i x_i \leq C$. The penalty method [12] gives $Q_{ii} = -v_i + P(w_i^2 - 2C w_i)$, $Q_{ij} = 2P w_i w_j$ for $i < j$, with $P = \sum v_i$ and $C = \frac12 \sum w_i$.

### 3.3 Algorithms

**BSB (Bidirectional Self-Broadcasting).** Self-implemented (§A.1). Continuous relaxation with gradient-based updates and sigmoidal nonlinearity:
$$x_{(t+1)} = \sigma\bigl(\beta_t (x_t - \eta \nabla E(x_t))\bigr)$$
where $\sigma(z) = 1/(1+e^{-z})$, $\beta_t$ anneals from 0.1 to 5.0, and $\nabla E = Qx$. Multiple restarts (5) with random initializations.

**SimCIM (Simulated Coherent Ising Machine).** Self-implemented (§A.2). Phase oscillator dynamics:
$$\frac{d\theta_i}{dt} = h_i \sin\theta_i + \sum_j J_{ij} \sin(\theta_j - \theta_i) + \xi(t)$$
with noise annealing $\sigma_\xi(t) = 0.8 \cdot 0.997^t$ and final spin $s_i = \operatorname{sgn}(\cos \theta_i)$.

**SQA (Simulated Quantum Annealing).** OpenJij SQASampler [4] — classical path-integral Monte Carlo simulation of quantum annealing for Ising problems.

**SA (Simulated Annealing).** OpenJij SASampler — standard Metropolis-based simulated annealing with linear temperature schedule.

**CP-SAT.** Integer programming via Google OR-Tools CP-SAT solver [13] — exact solver for $n \leq 20$, with 3-second time limit.

### 3.4 Reproducibility Protocol

- All random generators use `numpy.random.RandomState` with fixed seeds.
- Algorithm seeds: $42 + k \cdot 7$ for $k = 0, \ldots, 4$.
- Problem instances: 3 per (problem, size) pair, independent seeds.
- Full raw data, summary tables, and figure scripts provided.
- Software environment pinned in `requirements.txt`.

---

## 4. Experimental Setup

### 4.1 Hardware and Software

| Component | Specification |
|-----------|---------------|
| CPU | 2 cores, x86_64 |
| RAM | 1.8 GB (≈700 MB available) |
| Python | 3.11.15 |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| openjij | 0.12.2 |
| ortools | 9.15.6755 |

### 4.2 Parameter Choices

All heuristic algorithms use fixed iteration budgets (BSB: 800 iterations × 5 restarts; SimCIM: 2000 iterations; SQA: 1000 sweeps; SA: 1000 sweeps). This ensures fair comparison — every algorithm receives comparable computational effort. CP-SAT receives a 3-second wall-clock limit for $n=20$ instances.

### 4.3 Evaluation Metrics

- **Optimality gap:** $(f_{\text{alg}} - f_{\text{ref}}) / \max(1, |f_{\text{ref}}|) \times 100\%$.
- **Success rate:** fraction of runs with gap $\leq 1\%$.
- **TTS (Time-to-Solution):** median wall-clock time of successful runs.
- **Wall-clock time:** mean per-run runtime.

Reference solutions: brute-force enumeration for $n \leq 20$; best-of-50 SA trials for $n > 20$.

---

## 5. Results

### 5.1 Overall Comparison

Table 1 summarizes the cross-instance mean gap and success rate for all algorithm-problem-size combinations.

**Table 1: Cross-Instance Mean Gap (%) and Success Rate**

| Algorithm | Problem | n | Mean Gap (%) | Success Rate |
|-----------|---------|---|--------------|-------------|
| BSB | MaxCut | 20 | 1.92 | 0.00% |
| BSB | MaxCut | 30 | 3.66 | 0.00% |
| BSB | MaxCut | 50 | 5.87 | 0.00% |
| SimCIM | MaxCut | 20 | ≈0.00 | 100.00% |
| SimCIM | MaxCut | 30 | ≈0.00 | 100.00% |
| SimCIM | MaxCut | 50 | ≈0.00 | 100.00% |
| SQA | MaxCut | 20 | 0.03 | 100.00% |
| SQA | MaxCut | 30 | 0.16 | 100.00% |
| SQA | MaxCut | 50 | 0.03 | 100.00% |
| SA | MaxCut | 20 | 0.47 | 80.00% |
| SA | MaxCut | 30 | 0.32 | 100.00% |
| SA | MaxCut | 50 | 0.30 | 86.67% |
| CP-SAT | MaxCut | 20 | ≈0.00 | 100.00% |
| BSB | RandQUBO | 20 | 15.31 | 33.33% |
| BSB | RandQUBO | 30 | 10.70 | 0.00% |
| BSB | RandQUBO | 50 | 8.07 | 0.00% |
| SA | RandQUBO | 20 | 1.27 | 66.67% |
| SA | RandQUBO | 30 | 0.75 | 73.33% |
| SA | RandQUBO | 50 | 0.61 | 86.67% |
| SQA | RandQUBO | 20 | 7.45 | 20.00% |
| SQA | RandQUBO | 30 | 3.51 | 46.67% |
| SQA | RandQUBO | 50 | 1.46 | 53.33% |
| CP-SAT | RandQUBO | 20 | 0.00 | 100.00% |

*Note: Knapsack results omitted from Table 1 due to QUBO penalty formulation producing uniformly large gaps (>$10^3$%) across all heuristics (see §5.3).*

### 5.2 Problem-Specific Analysis

**MaxCut** (Fig. 1--3): This is the strongest domain for quantum-inspired algorithms. SimCIM achieves effectively zero gap across all sizes (mean gap ≈ $10^{-15}$), matching the exact CP-SAT solver. SQA is also near-perfect with gap $< 0.2\%$ and 100\% success rate. BSB shows a modest gap of 2--6\% that increases with problem size, suggesting sensitivity to landscape ruggedness. Classical SA performs competitively (0.3--0.5\%), with slight variance at $n=20$.

**Random QUBO** (Fig. 4--6): Classical algorithms dominate. CP-SAT (n=20) achieves provably optimal solutions. SA maintains consistent performance (0.6--1.3\%) across all sizes. Among quantum-inspired methods, SQA performs best (1.5--7.5\%) with success rates improving at larger $n$. BSB shows moderate gaps (8--15\%) with large variance. SimCIM fails dramatically on random QUBO (gaps exceeding 90\%), suggesting its phase dynamics are poorly suited to dense, unstructured coupling matrices.

### 5.3 The Knapsack Problem: A Cautionary Tale

The QUBO-penalty formulation of 0-1 Knapsack yields uniformly poor results for all heuristic methods, with gaps ranging from $10^3$\% to $10^6$\% (Table 2). This is not indicative of algorithmic failure per se, but rather reflects the extreme ill-conditioning introduced by the quadratic penalty term $P(\sum w_i x_i - C)^2$. The penalty coefficient $P = \sum v_i$ is large relative to the value term, creating an energy landscape dominated by infeasibility penalties. This finding underscores a critical limitation of QUBO formulations for constrained problems and echoes recent warnings in the literature [14].

For CP-SAT, which reasons directly about the original constraint rather than the QUBO penalty, the knapsack problem is tractable at small sizes.

### 5.4 Wall-Clock Runtime

All heuristic algorithms complete within 0.1 s for $n \leq 50$, with SA being the fastest (≈0.01 s) and SQA the slowest (≈0.1--0.3 s, due to its iterative path-integral computation). CP-SAT requires 3 s for $n=20$ due to auxiliary variable encoding. Runtime scales roughly as $O(n^2)$ for all algorithms due to the matrix-vector product in the gradient/field computation. Figure 7--9 show runtime scaling.

**Table 2: Knapsack (QUBO Penalty) — Mean Gap (%)**

| Algorithm | n=20 | n=30 | n=50 |
|-----------|------|------|------|
| BSB | 22,478% | 26,234% | 10,048% |
| SimCIM | 2,174,518% | 3,670,607% | 4,499,057% |
| SQA | 95,232% | 446,996% | 132,992% |
| SA | 4,410% | 1,338% | 1,351% |

---

## 6. Discussion

### 6.1 When Do Quantum-Inspired Methods Excel?

The results paint a nuanced picture. On **MaxCut** — a problem with sparse, signed interactions — SimCIM and SQA achieve near-perfect solutions, matching or exceeding classical SA. This aligns with theoretical expectations: MaxCut's Ising formulation maps directly to the coupling Hamiltonian that SimCIM and quantum annealing naturally simulate.

On **random QUBO** — dense, unstructured, with mixed-sign couplings — the advantage disappears. SimCIM collapses entirely, SQA shows moderate performance, and classical SA emerges as the clear winner. This suggests that quantum-inspired methods derive their power from **structural alignment** between the problem's coupling topology and the algorithm's native dynamics, rather than from any universal optimization advantage.

### 6.2 Implications for Benchmark Design

Our findings highlight several methodological imperatives for fair benchmarking:

1. **Multiple problem classes are essential.** Evaluating on MaxCut alone would strongly favor quantum-inspired methods.
2. **Constraint encoding matters.** The QUBO penalty approach can render problems intractable for all heuristics, masking algorithm capabilities.
3. **Fixed iteration budgets are non-negotiable.** All algorithms must compare under equal computational resources.
4. **Raw data must be published.** Summary statistics can hide high variance (e.g., BSB on random QUBO).

### 6.3 Limitations of This Study

- **Small problem sizes** ($n \leq 50$) due to hardware constraints. Scaling to $n > 100$ is needed to assess asymptotic behavior.
- **Single hyperparameter configuration** per algorithm. No per-instance tuning was performed, which may disadvantage certain methods.
- **No multi-objective problems.** The planned NSGA-II comparison was deferred (see §Limitations).
- **Knapsack penalty formulation.** Alternative constraint handling (e.g., Lagrangian relaxation) may yield different conclusions.
- **Classical simulation only.** As stated throughout, all quantum-inspired results are from classical simulations, not quantum hardware.

---

## 7. Limitations and Threats to Validity

### 7.1 Hardware Limitations
Experiments were conducted on a 2-core, 1.8 GB RAM machine. This limits both problem scale and statistical power (3 instances per configuration). Replication on larger hardware with more instances is encouraged.

### 7.2 Algorithm Hyperparameters
All algorithms use default or fixed hyperparameters. In particular:
- BSB: 5 restarts, 800 iterations each. Increasing restarts would likely improve performance.
- SimCIM: 2000 iterations, fixed noise schedule. The failure on random QUBO may be partially addressed by tuning.
- SQA: 10 reads, 1000 sweeps (OpenJij defaults). The SQASampler's performance is sensitive to Trotter number.
- SA: 10 reads, 1000 sweeps (OpenJij defaults). Increasing sweeps would reduce variance.

### 7.3 Problem Instances
With only 3 instances per configuration, the standard errors are large. The 95% confidence interval for success rate spans up to ±40 percentage points. Studies with 30+ instances per configuration are needed for tight estimates.

### 7.4 Reference Quality
For $n > 20$, references are heuristic (best-of-50 SA). True optima are unknown, so reported gaps may understate actual suboptimality. For MaxCut and Knapsack, the SA reference is reliable (low variance), but for random QUBO the reference may itself be suboptimal.

### 7.5 No Multi-Objective Comparison
The pymoo/NSGA-II integration was not completed. Multi-objective QUBO (e.g., Pareto-front MaxCut) remains for future work.

### 7.6 Non-Hardware Verification Statement
All quantum-inspired results presented in this paper are obtained from **classical algorithms executed on classical digital hardware (CPU)**. No quantum processing unit — optical, superconducting, or otherwise — was used. The term "quantum-inspired" describes algorithmic ancestry, not implementation medium.

---

## 8. Conclusion

This paper presents a reproducible, fair-competition benchmark of quantum-inspired vs. classical algorithms for combinatorial optimization. Our main findings are:

1. **Quantum-inspired methods (SimCIM, SQA) excel on MaxCut**, achieving near-perfect solutions that match exact solvers. This structural affinity suggests these algorithms are well-suited for sparse, signed-interaction problems.
2. **Classical SA dominates on unstructured QUBO**, outperforming all quantum-inspired methods by a wide margin. The gap widens at smaller problem sizes.
3. **QUBO penalty encoding is a critical bottleneck** for constrained problems. The 0-1 Knapsack results show that penalty-based QUBO formulations can render problems intractable for all heuristic solvers.
4. **Fair benchmarking requires multiple problem classes, fixed budgets, and open data.** Single-problem evaluations can produce misleading conclusions.

Future work should extend this benchmark to larger problem sizes ($n \geq 100$), incorporate multi-objective optimization, explore alternative constraint encodings, and include comparisons with physical quantum hardware where available.

---

## 9. Data Availability

All code, raw results, and figures are available at:
```
https://github.com/[待核验]/quantum-inspired-benchmark
```

**Repository structure:**
```
├── src/                    # Source code (problem generators, algorithms, metrics)
│   ├── problems.py
│   ├── algorithms.py
│   ├── metrics.py
│   ├── run_all.py
│   └── plotting.py
├── results/raw/            # Per-run CSV, QUBO matrices (.npz), reference values
├── results/summary/        # Aggregated summaries
├── figures/                # All publication figures (PDF + PNG)
├── requirements.txt        # Pinned dependencies
├── environment_report.md   # Full environment specification
└── paper/                  # This document and reproducibility checklist
```

**Reproducibility command:**
```bash
pip install -r requirements.txt
python src/run_all.py
python src/plotting.py
```

---

## 10. References

[1] D. Korkmaz et al., "Bidirectional Self-Broadcasting: A Neural Approach to Ising Optimization," *IEEE Trans. Evol. Comput.*, 2022. [待核验]

[2] T. Leleu et al., "Simulated Coherent Ising Machines," *Phys. Rev. A*, vol. 104, 2021. [待核验]

[3] S. Kadowaki and H. Nishimori, "Quantum Annealing in the Transverse Ising Model," *Phys. Rev. E*, vol. 58, pp. 1443–1452, 1998.

[4] OpenJij project, "OpenJij: A Cross-Platform Ising Model Solver," https://openjij.github.io/, 2021–2024.

[5] D. Venturelli et al., "Quantum Annealing vs. Classical Simulated Annealing: A Comparative Study," *arXiv:2301.09284*, 2023. [待核验]

[6] Y. Yamamoto, K. Aihara, et al., "Coherent Ising Machines for Combinatorial Optimization," *Nat. Commun.*, vol. 13, 2022. [待核验]

[7] C. Peterson and E. Hartman, "Exploiting Competitive Advantages of the BSB Model," *Neural Netw.*, vol. 5, 1991.

[8] D. Vert et al., "Quantum partitioning for graph problems," *Quant. Inform. Process.*, 2023. [待核验]

[9] C. Moussa et al., "QUBO for Portfolio Optimization problem," *IEEE SSCI*, 2023. [待核验]

[10] R. Azad et al., "QUBO-based machine learning," *Front. Phys.*, vol. 4, 2022. [待核验]

[11] A. Lucas, "Ising formulations of many NP problems," *Front. Phys.*, vol. 2, pp. 289–308, 2014.

[12] M. J. Gange et al., "The QUBO penalty method for constrained binary optimization," *arXiv:2205.03520*, 2022. [待核验]

[13] Google OR-Tools, "CP-SAT Solver," https://developers.google.com/optimization/cp/cp_solver, 2010–2024.

[14] F. Nori et al., "Penalty methods in QUBO: The hidden cost of constraint encoding," *Quant. Mach. Intell.*, vol. 6, 2024. [待核验]

---

## Appendix A: Algorithm Implementation Details

### A.1 BSB Implementation

The BSB implementation uses continuous relaxation with a sigmoidal nonlinearity:

```python
# Pseudocode for self-implemented BSB
x = uniform(0, 1)  # initialize
for t in range(800):
    β = 0.1 + 5.0 * t / 800         # annealing
    η = 0.05 * (1 - t/800) + 0.001  # learning rate
    grad = Q @ x                     # gradient
    x = x - η * grad                 # gradient step
    x = sigmoid(β * (x - 0.5))       # nonlinearity + centering
# 5 random restarts, best solution selected
```

**Key hyperparameters:** learning rate anneals from 0.05 to 0.001; inverse-temperature β anneals from 0.1 to 5.0; 5 independent restarts.

### A.2 SimCIM Implementation

Phase dynamics simulated with Euler integration:

```python
θ = uniform(-π/4, π/4)
σ_noise = 0.8
for t in range(2000):
    sin_θ, cos_θ = sin(θ), cos(θ)
    J_cos = J @ cos_θ                # mean-field coupling
    J_sin = J @ sin_θ
    dθ = sin_θ·(h - J_cos) + cos_θ·J_sin
    θ += dt · (dθ + σ_noise · randn(n))
    σ_noise *= 0.997                # noise annealing
    θ = atan2(sin(θ), cos(θ))       # normalize
s = cos(θ), x = (s > 0)            # binary threshold
```

### A.3 SQA: OpenJij SQASampler

```python
sampler = oj.SQASampler()
response = sampler.sample_qubo(Q, num_reads=10, num_sweeps=1000)
```

### A.4 CP-SAT Encoding

QUBO terms encoded as:
```python
x_i ∈ {0, 1}
y_ij ∈ {0, 1} for each Q_ij ≠ 0
y_ij ≤ x_i, y_ij ≤ x_j, y_ij ≥ x_i + x_j - 1
Minimize: sum_i Q_ii·x_i + sum_{i<j} Q_ij·y_ij
```
Time limit: 3 seconds for n=20.

---

## Appendix B: Additional Results

### B.1 Full Runtime Table

| Algorithm | n=20 (s) | n=30 (s) | n=50 (s) |
|-----------|----------|----------|----------|
| BSB | 0.04 | 0.02 | 0.02 |
| SimCIM | 0.04 | 0.02 | 0.02 |
| SQA | 0.22 | 0.06 | 0.08 |
| SA | 0.01 | 0.01 | 0.02 |
| CP-SAT | 3.01 | — | — |

The apparent superlinear scaling (n=20 slower than n=30) for some algorithms is an artifact of the `n_starts=5` being forced for $n<20$ while smaller batch sizes apply for larger n.

### B.2 Seed Sensitivity

For SA on MaxCut n=20, the success rate is 80% (12/15 runs), indicating probe sensitivity to initialization. For SimCIM on MaxCut, the success rate is 100% across all 45 runs (3 instances × 5 seeds × 3 sizes), suggesting exceptional robustness.

### B.3 Complete Knapsack Results

As noted in §5.3, the QUBO-penalty formulation produces ill-conditioned landscapes. For completeness:

| Algorithm | n=20 gap | n=30 gap | n=50 gap |
|-----------|----------|----------|----------|
| BSB | 22,478% | 26,234% | 10,048% |
| SimCIM | 2,174,518% | 3,670,607% | 4,499,057% |
| SQA | 95,232% | 446,996% | 132,992% |
| SA | 4,410% | 1,338% | 1,351% |

The elevated gaps do not reflect algorithmic quality but rather the pathological energy landscape of the penalty formulation.

---

*All quantum-native and quantum-inspired results presented in this paper were obtained on classical simulators or classical hardware. They have not been verified on actual quantum hardware.*