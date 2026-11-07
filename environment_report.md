# Environment Report — Quantum-Inspired vs Classical Benchmark

## Date
2026-11-07

## Hardware
- CPU: 2 cores, x86_64
- RAM: 1.8 GB total (≈1.1 GB used, ≈703 MB available)
- No GPU
- No quantum hardware

## Software
- OS: Linux (Alibaba Cloud Linux / CentOS-like)
- Python: 3.11.15 (CPython)
- pip: 24.0

## Installed Dependencies

| Package | Version | Source | Status |
|---------|---------|--------|--------|
| numpy | 2.4.6 | PyPI | ✅ |
| scipy | 1.17.1 | PyPI | ✅ |
| pandas | 2.3.3 | PyPI | ✅ |
| matplotlib | 3.10.9 | PyPI | ✅ |
| openjij | 0.12.2 | PyPI (Jij Inc.) | ✅ QAIA-compatible Sampler |
| ortools | 9.15.6755 | PyPI (Google) | ✅ CP-SAT exact solver |
| mindquantum | 0.12.0 | PyPI (MindSpore) | ✅ QAIA: BSB, SimCIM, LQA |
| pymoo | 0.6.2 | PyPI | ✅ NSGA-II (multi-objective) |
| dimod | 0.12.23 | PyPI (D-Wave) | ✅ QUBO utilities |

## QAIA Algorithm Availability

| Algorithm | Source | Status |
|-----------|--------|--------|
| BSB (Bidirectional Self-Broadcasting) | mindquantum.algorithm.qaia | ✅ |
| SimCIM (Simulated Coherent Ising Machine) | mindquantum.algorithm.qaia | ✅ |
| LQA (Linear Quantum Annealing) | mindquantum.algorithm.qaia | ✅ |
| Simulated Annealing | OpenJij SA | ✅ |
| CP-SAT Exact Solver | OR-Tools | ✅ |
| NSGA-II (Multi-Objective) | pymoo | ✅ |

## Fallback Decisions
- No fallback needed — primary QAIA stack (MindQuantum) installed successfully.
- Qiskit/PennyLane circuit-level experiments: **Skipped** (appendix-only, not required for main results). Machine too memory-constrained for circuit simulations.

## Memory Constraints
- Available RAM ≈ 700 MB after baseline.
- Problem sizes n≤50 keep QUBO matrices ≤ 2500 variables → fine.
- Each QAIA algorithm run is lightweight (matrix-vector ops).
- CP-SAT for n=50 may be memory-intensive; will set time limit.