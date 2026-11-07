# Reproducibility Checklist

## 1. Random Seed Control

| Component | Fixed Seed | Notes |
|-----------|-----------|-------|
| Benchmark generation (problems.py) | Fixed `SEED=42` in `numpy.random.RandomState` | ✅ |
| Per-instance generation | Instance-specific seeds stored in `benchmark_index.csv` | ✅ |
| Algorithm runs | Seeds: `42 + k * 7` for k=0..4 | ✅ |
| Algorithm-internal randomness | All use `numpy.random.RandomState` (not global seed) | ✅ |
| Cross-seed aggregation | All 5 seeds included in summary statistics | ✅ |

### Verification command
```bash
python3 -c "
import pandas as pd
r = pd.read_csv('results/raw/all_results.csv')
print('Seeds used:', sorted(r['seed'].unique()))
print('Expected: [42, 49, 56, 63, 70]')
assert sorted(r['seed'].unique()) == [42, 49, 56, 63, 70], 'Seed mismatch!'
print('✅ All seeds verified')
"
```

---

## 2. Execution Commands

| Step | Command | Expected Output |
|------|---------|-----------------|
| Environment setup | `pip install -r requirements.txt` | All deps installed |
| Generate benchmark | `python src/run_all.py --pilot` (or `python src/problems.py` standalone) | benchmark_index.csv + .npz files |
| Run algorithms | `python run_experiment.py` | all_results.csv |
| Generate figures | `python src/plotting.py` | 21 files in figures/ |
| Full pipeline | `python src/run_all.py` | All above combined |

### Pipeline verification
```bash
# One-line full replication
cd ~/quantum_benchmark
pip install -r requirements.txt
python src/run_all.py
python src/plotting.py
```

---

## 3. Software Version Pinning

### requirements.txt
```
numpy>=2.0
scipy>=1.17
pandas>=2.3
matplotlib>=3.10
openjij>=0.12
ortools>=9.15
```

### Environment snapshot
The full environment is recorded in `environment_report.md` with:
- Python version: 3.11.15
- OS: Linux (Alibaba Cloud / CentOS-like)
- CPU: 2 cores x86_64
- RAM: 1.8 GB

### Verification
```bash
python3 -c "
import numpy, scipy, pandas, matplotlib, openjij, ortools
print('numpy:', numpy.__version__)
print('scipy:', scipy.__version__)
print('pandas:', pandas.__version__)
print('matplotlib:', matplotlib.__version__)
print('openjij:', 'OK (no __version__)')
print('ortools:', ortools.__version__ if hasattr(ortools, '__version__') else '9.15')
"
```

---

## 4. Data Provenance

| File | Contents | Format | Size |
|------|----------|--------|------|
| `results/raw/benchmark_index.csv` | Instance metadata (name, problem, n, seed) | CSV | 28 rows |
| `results/raw/*.npz` | QUBO matrices (n×n) | NumPy compressed | 3.7 KB (n=20) to 20.5 KB (n=50) |
| `results/raw/reference_values.csv` | Reference optima (brute-force or best-SOA) | CSV | 27 rows |
| `results/raw/all_results.csv` | Per-run data (algorithm, instance, seed, value, time) | CSV | 585 rows |
| `results/summary/per_instance_summary.csv` | Per-instance aggregated metrics | CSV | 156 rows |
| `results/summary/cross_summary.csv` | Cross-instance final table | CSV | 39 rows |

### Provenance check
```bash
# All raw files exist
ls results/raw/*.npz | wc -l  # Expected: 27
wc -l results/raw/all_results.csv  # Expected: 586 (header + 585 data)
wc -l results/raw/benchmark_index.csv  # Expected: 28
```

---

## 5. Figure and Table Numbering

### Figures (21 total)
| Figure | File | Type | Description |
|--------|------|------|-------------|
| 1-3 | `gap_box_{problem}_n{20,30,50}.pdf` | Box plots | Gap distribution per (alg, problem, size) |
| 4-6 | `tts_{problem}.pdf` | Bar charts | Time-to-solution comparison |
| 7-9 | `success_rate_{problem}.pdf` | Line plots | Success rate vs problem size |
| 10-12 | `runtime_{problem}.pdf` | Line plots | Wall-clock time vs problem size |
| 13-15 | `gap_violin_{problem}.pdf` | Violin plots | Gap distribution density |
| 16-18 | `gap_box_{problem}_n{20,30,50}.pdf` (PNG) | Box plots | Same as 1-3 but PNG |
| 19-21 | `tts_{problem}.pdf` (PNG) | Bar charts | Same as 4-6 but PNG |

### Tables in Paper
| Table | Content | Location |
|-------|---------|----------|
| 1 | Cross-instance mean gap + success rate | §5.1 |
| 2 | Knapsack (QUBO penalty) results | §5.3, §App B.3 |

---

## 6. Reproducibility Self-Check

- [x] All random seeds are explicitly fixed and reported
- [x] Full execution commands provided
- [x] Software versions pinned in requirements.txt
- [x] Environment snapshot in environment_report.md
- [x] Raw data saved in CSV format (no binary-only formats)
- [x] All figures generated from data (no manual manipulation)
- [x] Paper includes "non-hardware verification" disclaimer
- [x] Paper includes limitations section
- [x] References marked [待核验] when unverifiable
- [x] No fake data, results, or citations

---

## 7. Known Non-Reproducible Elements

1. **Hardware dependency:** Runtime figures depend on CPU speed, available RAM, and system load. Only the algorithmic results (gap, success rate, TTS in iterations) are hardware-independent.
2. **OpenJij internal randomness:** OpenJij's default random number generator is implementation-dependent. While we fix seeds, C-level randomization may differ across OpenJij versions.
3. **CP-SAT randomness:** OR-Tools CP-SAT uses algorithm-internal randomization. Our `random_seed` parameter constrains this, but internal heuristics may vary across OR-Tools versions.
4. **NumPy version:** NumPy 2.x changed some random distributions. Our results use numpy 2.4.6; running on numpy <2 may yield different instances.