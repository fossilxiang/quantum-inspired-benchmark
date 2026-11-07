"""
Plotting: generates all figures for the paper.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY_DIR = os.path.join(ROOT, 'results', 'summary')
FIG_DIR = os.path.join(ROOT, 'figures')

# Colors
COLORS = {
    'BSB': '#E63946',
    'SimCIM': '#F4A261',
    'LQA': '#2A9D8F',
    'SA': '#264653',
    'CP-SAT': '#E9C46A',
}
MARKERS = {
    'BSB': 'o', 'SimCIM': 's', 'LQA': '^', 'SA': 'D', 'CP-SAT': 'v'
}


def setup_style():
    plt.rcParams.update({
        'figure.dpi': 150,
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'figure.figsize': (8, 5),
        'font.family': 'serif',
    })


def plot_gap_boxplot(cross_df: pd.DataFrame, save=True):
    """Box plot of optimality gap per (algorithm, problem, n)."""
    setup_style()
    
    for pb in cross_df['problem'].unique():
        for n_val in sorted(cross_df[cross_df['problem'] == pb]['n'].unique()):
            fig, ax = plt.subplots(figsize=(8, 5))
            sub = cross_df[(cross_df['problem'] == pb) & (cross_df['n'] == n_val)]
            
            # We need per-instance data; load from summary
            summary = pd.read_csv(os.path.join(SUMMARY_DIR, 'per_instance_summary.csv'))
            sub_s = summary[(summary['problem'] == pb) & (summary['n'] == n_val)]
            
            algs = sorted(sub_s['algorithm'].unique())
            data = [sub_s[sub_s['algorithm'] == a]['mean_gap'].values for a in algs]
            
            bp = ax.boxplot(data, labels=algs, patch_artist=True)
            for patch, alg in zip(bp['boxes'], algs):
                if alg in COLORS:
                    patch.set_facecolor(COLORS[alg])
                    patch.set_alpha(0.7)
            
            ax.set_ylabel('Optimality Gap (%)')
            ax.set_title(f'Problem: {pb.upper()} | Size n={n_val}')
            ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5,
                       label='1% threshold')
            ax.legend(loc='best')
            ax.set_yscale('symlog', linthresh=0.1)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save:
                fname = f"gap_box_{pb}_n{n_val}.pdf"
                plt.savefig(os.path.join(FIG_DIR, fname), bbox_inches='tight')
                plt.savefig(os.path.join(FIG_DIR, fname.replace('.pdf', '.png')),
                           bbox_inches='tight', dpi=150)
                print(f"Saved {fname}")
            plt.close()


def plot_tts(cross_df: pd.DataFrame, save=True):
    """TTS comparison bar chart per problem/size."""
    setup_style()
    
    for pb in cross_df['problem'].unique():
        fig, ax = plt.subplots(figsize=(8, 5))
        sub = cross_df[cross_df['problem'] == pb]
        
        algs = sorted(sub['algorithm'].unique())
        sizes = sorted(sub['n'].unique())
        
        x = np.arange(len(sizes))
        width = 0.8 / len(algs)
        
        for i, alg in enumerate(algs):
            vals = []
            for n_val in sizes:
                row = sub[(sub['algorithm'] == alg) & (sub['n'] == n_val)]
                vals.append(row['tts_median'].values[0] if len(row) > 0 else 0)
            offset = (i - len(algs)/2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=alg,
                         color=COLORS.get(alg, '#888888'), alpha=0.85)
        
        ax.set_xticks(x)
        ax.set_xticklabels([f'n={n}' for n in sizes])
        ax.set_ylabel('Median TTS (s)')
        ax.set_title(f'Time-to-Solution: {pb.upper()}')
        ax.legend(loc='upper left')
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save:
            fname = f"tts_{pb}.pdf"
            plt.savefig(os.path.join(FIG_DIR, fname), bbox_inches='tight')
            plt.savefig(os.path.join(FIG_DIR, fname.replace('.pdf', '.png')),
                       bbox_inches='tight', dpi=150)
            print(f"Saved {fname}")
        plt.close()


def plot_success_rate(cross_df: pd.DataFrame, save=True):
    """Success rate vs problem size."""
    setup_style()
    
    for pb in cross_df['problem'].unique():
        fig, ax = plt.subplots(figsize=(8, 5))
        sub = cross_df[cross_df['problem'] == pb]
        
        for alg in sorted(sub['algorithm'].unique()):
            alg_sub = sub[sub['algorithm'] == alg].sort_values('n')
            ax.plot(alg_sub['n'], alg_sub['success_rate'], 
                   marker=MARKERS.get(alg, 'o'), label=alg,
                   color=COLORS.get(alg, '#888888'), linewidth=2,
                   markersize=8)
        
        ax.set_xlabel('Problem Size (n)')
        ax.set_ylabel('Success Rate (<1% gap)')
        ax.set_title(f'Success Rate vs Size: {pb.upper()}')
        ax.legend(loc='best')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            fname = f"success_rate_{pb}.pdf"
            plt.savefig(os.path.join(FIG_DIR, fname), bbox_inches='tight')
            plt.savefig(os.path.join(FIG_DIR, fname.replace('.pdf', '.png')),
                       bbox_inches='tight', dpi=150)
            print(f"Saved {fname}")
        plt.close()


def plot_runtime(cross_df: pd.DataFrame, save=True):
    """Mean wall-clock time vs problem size."""
    setup_style()
    
    for pb in cross_df['problem'].unique():
        fig, ax = plt.subplots(figsize=(8, 5))
        sub = cross_df[cross_df['problem'] == pb]
        
        for alg in sorted(sub['algorithm'].unique()):
            alg_sub = sub[sub['algorithm'] == alg].sort_values('n')
            ax.plot(alg_sub['n'], alg_sub['mean_time'], 
                   marker=MARKERS.get(alg, 'o'), label=alg,
                   color=COLORS.get(alg, '#888888'), linewidth=2,
                   markersize=8)
        
        ax.set_xlabel('Problem Size (n)')
        ax.set_ylabel('Mean Wall-clock Time (s)')
        ax.set_title(f'Runtime vs Size: {pb.upper()}')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            fname = f"runtime_{pb}.pdf"
            plt.savefig(os.path.join(FIG_DIR, fname), bbox_inches='tight')
            plt.savefig(os.path.join(FIG_DIR, fname.replace('.pdf', '.png')),
                       bbox_inches='tight', dpi=150)
            print(f"Saved {fname}")
        plt.close()


def plot_gap_violin_by_algorithm(cross_df: pd.DataFrame, save=True):
    """Violin plot of gap distributions across all instances per algorithm."""
    setup_style()
    summary = pd.read_csv(os.path.join(SUMMARY_DIR, 'per_instance_summary.csv'))
    
    for pb in cross_df['problem'].unique():
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
        sizes = sorted(cross_df[cross_df['problem'] == pb]['n'].unique())
        
        for idx, n_val in enumerate(sizes):
            ax = axes[idx]
            sub = summary[(summary['problem'] == pb) & (summary['n'] == n_val)]
            algs = sorted(sub['algorithm'].unique())
            
            data = [sub[sub['algorithm'] == a]['mean_gap'].values for a in algs]
            parts = ax.violinplot(data, showmeans=True, showmedians=True)
            for pc, alg in zip(parts['bodies'], algs):
                pc.set_facecolor(COLORS.get(alg, '#888888'))
                pc.set_alpha(0.7)
            
            ax.set_xticks(range(1, len(algs)+1))
            ax.set_xticklabels(algs, fontsize=8)
            ax.set_title(f'n={n_val}')
            ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
            ax.grid(True, axis='y', alpha=0.3)
        
        axes[0].set_ylabel('Optimality Gap (%)')
        fig.suptitle(f'Gap Distribution: {pb.upper()}', fontsize=14)
        plt.yscale('symlog', linthresh=0.1)
        plt.tight_layout()
        
        if save:
            fname = f"gap_violin_{pb}.pdf"
            plt.savefig(os.path.join(FIG_DIR, fname), bbox_inches='tight')
            plt.savefig(os.path.join(FIG_DIR, fname.replace('.pdf', '.png')),
                       bbox_inches='tight', dpi=150)
            print(f"Saved {fname}")
        plt.close()


def generate_all_plots():
    """Generate all figures from saved summary data."""
    os.makedirs(FIG_DIR, exist_ok=True)
    
    cross_file = os.path.join(SUMMARY_DIR, 'cross_summary.csv')
    if not os.path.exists(cross_file):
        print("No cross_summary.csv found. Run experiment first.")
        return
    
    cross_df = pd.read_csv(cross_file)
    
    print("Generating figures...")
    plot_gap_boxplot(cross_df)
    plot_tts(cross_df)
    plot_success_rate(cross_df)
    plot_runtime(cross_df)
    plot_gap_violin_by_algorithm(cross_df)
    print(f"All figures saved to {FIG_DIR}/")


if __name__ == '__main__':
    generate_all_plots()