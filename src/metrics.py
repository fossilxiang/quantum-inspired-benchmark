"""
Metrics computation: optimality gap, success rate, TTS, summary statistics.
"""

import numpy as np
import pandas as pd


def compute_gap(found: float, reference: float) -> float:
    """Relative gap in %, handling zero reference."""
    denom = max(1.0, abs(reference))
    return float((found - reference) / denom * 100.0)


def compute_success(gap: float, threshold: float = 1.0) -> bool:
    """Success if gap is within threshold %."""
    return gap <= threshold


def compute_tts(times: list, successes: list) -> float:
    """Time-to-Solution: median time of successful runs, or inf if none."""
    succ_times = [t for t, s in zip(times, successes) if s]
    if not succ_times:
        return float('inf')
    return float(np.median(succ_times))


def summarize_results(df: pd.DataFrame, reference_col: str = 'reference_value') -> pd.DataFrame:
    """Aggregate per (algorithm, problem, n, instance).
    
    Input columns: algorithm, problem, n, instance, seed, value, time, reference_value
    Returns: summary with mean_gap, median_gap, success_rate, tts, mean_time, std_time
    """
    df['gap'] = df.apply(
        lambda r: compute_gap(r['value'], r[reference_col]), axis=1)
    df['success'] = df['gap'].apply(lambda g: compute_success(g, 1.0))
    
    groups = df.groupby(['algorithm', 'problem', 'n', 'instance'])
    
    summary = groups.agg(
        mean_gap=('gap', 'mean'),
        median_gap=('gap', 'median'),
        std_gap=('gap', 'std'),
        success_rate=('success', 'mean'),
        mean_time=('time', 'mean'),
        std_time=('time', 'std'),
        n_runs=('gap', 'count'),
    ).reset_index()
    
    # TTS: time-to-solution (median time of successful runs)
    tts_data = []
    for (alg, pb, n_val, inst), grp in groups:
        succ_times = grp[grp['success']]['time'].values
        tts = float(np.median(succ_times)) if len(succ_times) > 0 else float('inf')
        tts_data.append({'algorithm': alg, 'problem': pb, 'n': n_val,
                         'instance': inst, 'tts': tts})
    
    tts_df = pd.DataFrame(tts_data)
    summary = summary.merge(tts_df, on=['algorithm', 'problem', 'n', 'instance'])
    
    return summary


def cross_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-instance aggregation: final table for paper."""
    agg = summary_df.groupby(['algorithm', 'problem', 'n']).agg(
        mean_gap=('mean_gap', 'mean'),
        std_gap_across=('mean_gap', 'std'),
        success_rate=('success_rate', 'mean'),
        tts_median=('tts', lambda x: float(np.median(x))),
        mean_time=('mean_time', 'mean'),
    ).reset_index()
    
    return agg


if __name__ == '__main__':
    # Quick test
    print("Metrics module loaded. compute_gap(10, 5) =", compute_gap(10, 5))