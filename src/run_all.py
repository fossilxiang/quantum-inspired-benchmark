"""
Experiment runner: orchestrates benchmark generation, algorithm execution,
result collection, and summary.

Usage:
    python run_all.py [--pilot] [--sizes 20 30] [--n-instances 5] [--n-seeds 5]
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from problems import generate_benchmark, evaluate_qubo, brute_force_qubo
from algorithms import get_solver_names, run_solver
from metrics import summarize_results, cross_summary


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, 'results', 'raw')
SUMMARY_DIR = os.path.join(ROOT, 'results', 'summary')


def get_reference(Q, offset, n):
    """Get reference optimum for an instance."""
    if n <= 20:
        x_opt, val_opt = brute_force_qubo(Q, offset)
        return float(val_opt), "brute_force"
    else:
        # Use CP-SAT with generous time limit as reference
        res = run_solver("CP-SAT", Q, offset, time_limit=60.0, seed=42)
        # Also run many SA trials to get best
        best_val = res['value']
        # Run SA with many seeds
        for s in range(50):
            sa_res = run_solver("SA", Q, offset, time_limit=5.0, seed=42+s)
            if sa_res['value'] < best_val:
                best_val = sa_res['value']
        return float(best_val), "CP-SAT+SA_best"


def run_experiment(sizes=(20, 30), n_instances=3, n_seeds=5,
                   problems=('random_qubo', 'maxcut', 'knapsack'),
                   algorithms=None, time_limit=30.0, pilot=False):
    """Main experiment loop."""
    
    if algorithms is None:
        algorithms = get_solver_names()
    
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    
    # Step 1: Generate benchmark instances (only if not already done)
    index_file = os.path.join(RAW_DIR, 'benchmark_index.csv')
    if not os.path.exists(index_file):
        print("=== Generating benchmark instances ===")
        generate_benchmark(sizes=sizes, n_instances=n_instances,
                          problems=problems, output_dir=RAW_DIR)
    
    # Load index
    df_idx = pd.read_csv(index_file)
    print(f"Loaded {len(df_idx)} instances from index")
    
    # Step 2: For each instance, compute reference solution
    # Check if reference already exists
    ref_file = os.path.join(RAW_DIR, 'reference_values.csv')
    if os.path.exists(ref_file):
        ref_df = pd.read_csv(ref_file)
        ref_dict = {(r['name']): r['reference_value'] 
                     for _, r in ref_df.iterrows()}
        print(f"Loaded {len(ref_dict)} reference values")
    else:
        print("=== Computing reference solutions ===")
        ref_dict = {}
        ref_rows = []
        for _, row in df_idx.iterrows():
            fname = row['name']
            data = np.load(os.path.join(RAW_DIR, f"{fname}.npz"))
            Q = data['Q']
            offset = float(data['offset'])
            n = data['Q'].shape[0]
            ref_val, method = get_reference(Q, offset, n)
            ref_dict[fname] = ref_val
            ref_rows.append({"name": fname, "reference_value": ref_val,
                            "reference_method": method})
            print(f"  {fname}: ref={ref_val:.4f} ({method})")
        
        pd.DataFrame(ref_rows).to_csv(ref_file, index=False)
        print(f"Reference values saved to {ref_file}")
    
    # Step 3: Run algorithms
    all_results = []
    
    for _, row in df_idx.iterrows():
        fname = row['name']
        n = row['n']
        pb = row['problem']
        
        data = np.load(os.path.join(RAW_DIR, f"{fname}.npz"))
        Q = data['Q']
        offset = float(data['offset'])
        ref_val = ref_dict[fname]
        
        for alg in algorithms:
            if alg == "CP-SAT" and n > 30 and not pilot:
                # Skip CP-SAT for large instances (too expensive)
                print(f"  Skipping CP-SAT for {fname} (n={n} too large)")
                continue
            
            for seed in range(n_seeds):
                rng_seed = 42 + seed * 7
                
                result = run_solver(alg, Q, offset, 
                                   time_limit=time_limit, seed=rng_seed)
                
                row_data = {
                    'algorithm': alg,
                    'problem': pb,
                    'n': n,
                    'instance': row['instance'],
                    'seed': rng_seed,
                    'value': result['value'],
                    'time': result['time'],
                    'reference_value': ref_val,
                }
                all_results.append(row_data)
                print(f"  {alg:10s} {fname:30s} seed={rng_seed:4d} "
                      f"val={result['value']:.4f} [{result['time']:.3f}s]")
    
    # Save raw results
    results_df = pd.DataFrame(all_results)
    results_file = os.path.join(RAW_DIR, 'all_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"\nRaw results saved: {results_file} ({len(results_df)} rows)")
    
    # Step 4: Compute summaries
    print("\n=== Computing summaries ===")
    summary = summarize_results(results_df)
    summary_file = os.path.join(SUMMARY_DIR, 'per_instance_summary.csv')
    summary.to_csv(summary_file, index=False)
    print(f"Per-instance summary: {summary_file}")
    
    cross = cross_summary(summary)
    cross_file = os.path.join(SUMMARY_DIR, 'cross_summary.csv')
    cross.to_csv(cross_file, index=False)
    print(f"Cross summary: {cross_file}")
    print(cross.to_string())
    
    return results_df, summary, cross


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true', 
                       help='Run pilot (small) experiment')
    parser.add_argument('--sizes', type=int, nargs='+', default=[20, 30, 50])
    parser.add_argument('--n-instances', type=int, default=10)
    parser.add_argument('--n-seeds', type=int, default=10)
    parser.add_argument('--time-limit', type=float, default=30.0)
    parser.add_argument('--no-cpsat', action='store_true',
                       help='Skip CP-SAT for all sizes')
    args = parser.parse_args()
    
    if args.pilot:
        print("=== PILOT MODE ===")
        sizes = (20,)
        n_instances = 2
        n_seeds = 3
        problems = ('random_qubo', 'maxcut')
        algs = ['BSB', 'SA']
        time_limit = 10.0
    else:
        print("=== FULL EXPERIMENT ===")
        sizes = tuple(args.sizes)
        n_instances = args.n_instances
        n_seeds = args.n_seeds
        problems = ('random_qubo', 'maxcut', 'knapsack')
        algs = get_solver_names()
        if args.no_cpsat:
            algs = [a for a in algs if a != 'CP-SAT']
        time_limit = args.time_limit
    
    results_df, summary, cross = run_experiment(
        sizes=sizes, n_instances=n_instances, n_seeds=n_seeds,
        problems=problems, algorithms=algs, time_limit=time_limit,
        pilot=args.pilot,
    )
    
    print("\n=== DONE ===")


if __name__ == '__main__':
    main()