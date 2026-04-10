#!/usr/bin/env python3
"""Stage B2 + B3 comprehensive analysis.

Compares five training regimes:
  - selective:         delta-based top-k mask from curated reference (original Phase 3)
  - full_baseline:     all speech tokens (B1)
  - random_mask:       random 60% mask, same 807 utts (B2)
  - random_ref:        delta from random-reference model, all 1000 utts (B3 unfair)
  - random_ref_fair:   delta from random-reference model, same 807 utts (B3 fair)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Fair comparison set (same 807-utt mask coverage)
FAIR_RUNS = ['selective', 'full_baseline', 'random_mask', 'random_ref_fair']
ALL_RUNS = ['selective', 'full_baseline', 'random_mask', 'random_ref', 'random_ref_fair']

COLORS = {
    'selective':        '#d62728',
    'full_baseline':    '#1f77b4',
    'random_mask':      '#2ca02c',
    'random_ref':       '#ff7f0e',
    'random_ref_fair':  '#9467bd',
}
LABELS = {
    'selective':        'Selective (delta, curated ref)',
    'full_baseline':    'Full baseline',
    'random_mask':      'Random mask (same ratio)',
    'random_ref':       'Random ref (1000 utts, unfair)',
    'random_ref_fair':  'Random ref (807 utts, fair)',
}


def load_csv(path):
    df = pd.read_csv(path)
    df.columns = ['wall_time', 'step', 'value']
    return df

def deduplicate_runs(df):
    if len(df) == 0:
        return df
    step_diff = df['step'].diff()
    run_starts = [0] + list(df.index[step_diff < -5]) + [len(df)]
    if len(run_starts) <= 2:
        return df
    last_start = run_starts[-2]
    return df.iloc[last_start:].reset_index(drop=True)

def compute_timing(df):
    if len(df) < 2:
        return {}
    total_time = df['wall_time'].iloc[-1] - df['wall_time'].iloc[0]
    n_steps = df['step'].iloc[-1] - df['step'].iloc[0]
    return {
        'total_wall_seconds': float(total_time),
        'total_steps': int(n_steps),
        'seconds_per_step': float(total_time / n_steps) if n_steps > 0 else 0,
    }


def main():
    print("=" * 72)
    print("Stage B2 + B3: Complete Comparative Analysis")
    print("=" * 72)

    data = {}
    for run in ALL_RUNS:
        train_path = OUTPUT_DIR / f"{run}_TRAIN_loss.csv"
        cv_path = OUTPUT_DIR / f"{run}_CV_loss.csv"
        train = deduplicate_runs(load_csv(train_path))
        cv = deduplicate_runs(load_csv(cv_path))
        data[run] = {'train': train, 'cv': cv, 'timing': compute_timing(train)}

    train_first = {r: float(data[r]['train']['value'].iloc[0]) for r in ALL_RUNS}
    train_final = {r: float(data[r]['train']['value'].iloc[-1]) for r in ALL_RUNS}
    cv_finals = {r: float(data[r]['cv']['value'].iloc[-1]) for r in ALL_RUNS}
    cv_by_ep = {r: data[r]['cv']['value'].tolist() for r in ALL_RUNS}

    # ══════════════════════════════════════════════════════════════
    # 1. FAIR COMPARISON (same 807-utt coverage)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("SECTION 1: Fair Comparison (same 807-utt mask coverage)")
    print("=" * 72)

    print(f"\n{'Run':<35} {'Train Final':>12} {'CV Final':>12}")
    print("-" * 60)
    for run in FAIR_RUNS:
        print(f"{LABELS[run]:<35} {train_final[run]:>12.4f} {cv_finals[run]:>12.4f}")

    fair_ranked = sorted([(r, cv_finals[r]) for r in FAIR_RUNS], key=lambda x: x[1])
    print(f"\nFair CV ranking:")
    for i, (r, v) in enumerate(fair_ranked, 1):
        marker = " ★" if i == 1 else ""
        print(f"  {i}. {LABELS[r]:<35} {v:.4f}{marker}")

    print("\n--- B2 analysis (selective vs random_mask) ---")
    d_cv = cv_finals['selective'] - cv_finals['random_mask']
    d_tr = train_final['selective'] - train_final['random_mask']
    print(f"  CV diff:    {d_cv:+.4f}  ({'selective worse' if d_cv > 0 else 'selective better'})")
    print(f"  Train diff: {d_tr:+.4f}  ({'selective worse' if d_tr > 0 else 'selective better'})")
    b2_conclusion = 'random mask' if d_cv > 0 else 'selective'
    print(f"  → B2 winner (CV): {b2_conclusion}")

    print("\n--- B3 analysis (selective vs random_ref_fair) ---")
    d_cv = cv_finals['selective'] - cv_finals['random_ref_fair']
    d_tr = train_final['selective'] - train_final['random_ref_fair']
    print(f"  CV diff:    {d_cv:+.4f}  ({'selective worse' if d_cv > 0 else 'selective better'})")
    print(f"  Train diff: {d_tr:+.4f}  ({'selective worse' if d_tr > 0 else 'selective better'})")
    b3_conclusion = 'random_ref' if d_cv > 0 else 'selective'
    print(f"  → B3 winner (CV): {b3_conclusion}")

    print("\n--- B3 confound analysis (1000 vs 807 utts) ---")
    d_unfair = cv_finals['random_ref'] - cv_finals['random_ref_fair']
    print(f"  random_ref (1000 utts): {cv_finals['random_ref']:.4f}")
    print(f"  random_ref_fair (807):  {cv_finals['random_ref_fair']:.4f}")
    print(f"  Impact of extra 193 utts: {d_unfair:+.4f}")
    print(f"  → Mask coverage confound explains {abs(d_unfair):.4f} of the {abs(cv_finals['selective'] - cv_finals['random_ref']):.4f} gap")

    # ══════════════════════════════════════════════════════════════
    # 2. CV LOSS PROGRESSION TABLE
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("SECTION 2: CV Loss Progression by Epoch")
    print("=" * 72)
    print(f"{'Run':<35} {'Ep0':>10} {'Ep1':>10} {'Ep2':>10}")
    print("-" * 66)
    for run in ALL_RUNS:
        vals = cv_by_ep[run]
        vals_str = " ".join(f"{v:>10.4f}" for v in vals[:3])
        print(f"{LABELS[run]:<35} {vals_str}")

    # ══════════════════════════════════════════════════════════════
    # 3. WALL-CLOCK
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("SECTION 3: Wall-Clock Timing")
    print("=" * 72)
    print(f"{'Run':<35} {'Total (s)':>10} {'Per step':>10}")
    print("-" * 56)
    for run in ALL_RUNS:
        t = data[run]['timing']
        print(f"{LABELS[run]:<35} {t.get('total_wall_seconds', 0):>10.1f} {t.get('seconds_per_step', 0):>10.3f}")

    # ══════════════════════════════════════════════════════════════
    # Figures
    # ══════════════════════════════════════════════════════════════
    print("\nGenerating figures...")

    # Fig 1: Fair 4-way comparison - train + CV
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    window = 10
    ax = axes[0]
    for run in FAIR_RUNS:
        df = data[run]['train']
        smooth = df['value'].rolling(window, min_periods=1).mean()
        ax.plot(df['step'], df['value'], alpha=0.15, color=COLORS[run])
        ax.plot(df['step'], smooth, color=COLORS[run], linewidth=2,
                label=f"{LABELS[run]}")
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Train Loss', fontsize=12)
    ax.set_title('Train Loss (fair comparison)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for run in FAIR_RUNS:
        df = data[run]['cv']
        ax.plot(df['step'], df['value'], '-o', color=COLORS[run], linewidth=2.2,
                markersize=9, label=f"{LABELS[run]}")
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('CV Loss', fontsize=12)
    ax.set_title('CV Loss (fair comparison)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.suptitle('Stage B: Fair 4-way Comparison (all with 807-utt coverage)', fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'b23_fair_comparison.png', dpi=150)
    plt.close(fig)

    # Fig 2: Final CV bar chart (fair)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(FAIR_RUNS))
    vals = [cv_finals[r] for r in FAIR_RUNS]
    bars = ax.bar(x, vals, color=[COLORS[r] for r in FAIR_RUNS], edgecolor='black', alpha=0.85, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[r] for r in FAIR_RUNS], rotation=18, ha='right', fontsize=9.5)
    ax.set_ylabel('Final CV Loss', fontsize=12)
    ax.set_title('Final CV Loss (Fair Comparison, 807-utt coverage)', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    ymin = min(vals) - 0.15
    ax.set_ylim(ymin, max(vals) + 0.08)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'b23_fair_final_cv.png', dpi=150)
    plt.close(fig)

    # Fig 3: B2 focus
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    for run in ['selective', 'random_mask']:
        df = data[run]['train']
        smooth = df['value'].rolling(window, min_periods=1).mean()
        ax.plot(df['step'], df['value'], alpha=0.2, color=COLORS[run])
        ax.plot(df['step'], smooth, color=COLORS[run], linewidth=2.5,
                label=f"{LABELS[run]} (final={train_final[run]:.4f})")
    ax.set_xlabel('Step')
    ax.set_ylabel('Train Loss')
    ax.set_title('B2: Train Loss')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for run in ['selective', 'random_mask']:
        df = data[run]['cv']
        ax.plot(df['step'], df['value'], '-o', color=COLORS[run], linewidth=2.5,
                markersize=10, label=f"{LABELS[run]} (final={cv_finals[run]:.4f})")
    ax.set_xlabel('Step')
    ax.set_ylabel('CV Loss')
    ax.set_title('B2: CV Loss')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.suptitle('Stage B2: Delta-based Selection vs Random Selection', fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'b2_selective_vs_random_mask.png', dpi=150)
    plt.close(fig)

    # Fig 4: B3 focus (fair)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    for run in ['selective', 'random_ref_fair']:
        df = data[run]['train']
        smooth = df['value'].rolling(window, min_periods=1).mean()
        ax.plot(df['step'], df['value'], alpha=0.2, color=COLORS[run])
        ax.plot(df['step'], smooth, color=COLORS[run], linewidth=2.5,
                label=f"{LABELS[run]} (final={train_final[run]:.4f})")
    ax.set_xlabel('Step')
    ax.set_ylabel('Train Loss')
    ax.set_title('B3: Train Loss')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for run in ['selective', 'random_ref_fair']:
        df = data[run]['cv']
        ax.plot(df['step'], df['value'], '-o', color=COLORS[run], linewidth=2.5,
                markersize=10, label=f"{LABELS[run]} (final={cv_finals[run]:.4f})")
    ax.set_xlabel('Step')
    ax.set_ylabel('CV Loss')
    ax.set_title('B3: CV Loss')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.suptitle('Stage B3: Curated Reference vs Random Reference', fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'b3_selective_vs_random_ref_fair.png', dpi=150)
    plt.close(fig)

    # Fig 5: B3 confound (1000 vs 807 utts)
    fig, ax = plt.subplots(figsize=(10, 6))
    for run in ['selective', 'random_ref', 'random_ref_fair']:
        df = data[run]['cv']
        ax.plot(df['step'], df['value'], '-o', color=COLORS[run], linewidth=2.5,
                markersize=10, label=f"{LABELS[run]} (final={cv_finals[run]:.4f})")
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('CV Loss', fontsize=12)
    ax.set_title('B3 Confound: Impact of Mask Coverage (1000 vs 807 utts)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'b3_confound_coverage.png', dpi=150)
    plt.close(fig)

    # Fig 6: All 5 runs overview
    fig, ax = plt.subplots(figsize=(12, 7))
    for run in ALL_RUNS:
        df = data[run]['cv']
        ls = '--' if run == 'random_ref' else '-'
        ax.plot(df['step'], df['value'], ls, marker='o', color=COLORS[run], linewidth=2.3,
                markersize=9, label=LABELS[run])
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('CV Loss', fontsize=12)
    ax.set_title('All Stage B Experiments — CV Loss Progression', fontsize=14)
    ax.legend(fontsize=9.5, loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'b23_all_runs_cv.png', dpi=150)
    plt.close(fig)

    print(f"All figures saved to {FIGURES_DIR}")

    # ── Save summary ──
    summary = {
        'runs': {
            run: {
                'train_final': train_final[run],
                'cv_final': cv_finals[run],
                'cv_by_epoch': cv_by_ep[run],
                'timing': data[run]['timing'],
            } for run in ALL_RUNS
        },
        'fair_ranking': [{'run': r, 'label': LABELS[r], 'cv_final': v} for r, v in fair_ranked],
        'b2_selective_vs_random_mask': {
            'cv_diff': float(cv_finals['selective'] - cv_finals['random_mask']),
            'train_diff': float(train_final['selective'] - train_final['random_mask']),
            'winner': b2_conclusion,
        },
        'b3_selective_vs_random_ref_fair': {
            'cv_diff': float(cv_finals['selective'] - cv_finals['random_ref_fair']),
            'train_diff': float(train_final['selective'] - train_final['random_ref_fair']),
            'winner': b3_conclusion,
        },
        'b3_coverage_confound': {
            'random_ref_1000': cv_finals['random_ref'],
            'random_ref_fair_807': cv_finals['random_ref_fair'],
            'coverage_effect': float(cv_finals['random_ref'] - cv_finals['random_ref_fair']),
        },
    }
    summary_path = OUTPUT_DIR / 'stage_b23_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == '__main__':
    main()
