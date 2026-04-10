#!/usr/bin/env python3
"""Stage B1: Selective vs Full Training Baseline comparison.

Corresponds to next_steps_experiment_analysis_plan.md — Stage B1:
  Compare delta-based selective training against standard full training.

Data sources:
  - selective:   phase3_outputs/tensorboard (exported CSV)
  - full_phase3: phase3_full2_outputs/tensorboard (exported CSV)

Usage:
    python stage_b_analysis/run_analysis.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# CSV files (TensorBoard exports)
FULL_TRAIN_CSV = OUTPUT_DIR / "full_phase3_train-loss.csv"
FULL_CV_CSV = OUTPUT_DIR / "full_phase3_..csv"       # CV loss
SEL_TRAIN_CSV = OUTPUT_DIR / "selective_train-loss.csv"
SEL_CV_CSV = OUTPUT_DIR / "selective_..csv"           # CV loss


def load_csv(path, label=""):
    """Load TensorBoard-exported CSV and return DataFrame with Wall time, Step, Value."""
    df = pd.read_csv(path)
    df.columns = ['wall_time', 'step', 'value']
    print(f"  [{label}] Loaded {len(df)} rows from {path.name}")
    return df


def deduplicate_runs(df):
    """If CSV contains multiple runs (different timestamp epochs),
    keep only the LAST complete run (highest wall_time epoch)."""
    if len(df) == 0:
        return df

    # Detect run boundaries: large backward jumps in step
    step_diff = df['step'].diff()
    # A new run starts where step decreases (e.g., goes from 66 back to 1)
    run_starts = [0] + list(df.index[step_diff < -5]) + [len(df)]

    if len(run_starts) <= 2:
        return df  # single run

    # Take the last run
    last_start = run_starts[-2]
    result = df.iloc[last_start:].reset_index(drop=True)
    n_runs = len(run_starts) - 1
    print(f"    -> Found {n_runs} runs in CSV, keeping last run ({len(result)} rows)")
    return result


def compute_timing(df):
    """Compute timing info from wall_time column."""
    if len(df) < 2:
        return {}
    total_time = df['wall_time'].iloc[-1] - df['wall_time'].iloc[0]
    n_steps = df['step'].iloc[-1] - df['step'].iloc[0]
    return {
        'total_wall_seconds': float(total_time),
        'total_steps': int(n_steps),
        'seconds_per_step': float(total_time / n_steps) if n_steps > 0 else 0,
        'first_step': int(df['step'].iloc[0]),
        'last_step': int(df['step'].iloc[-1]),
    }


def interpolate_at_steps(df, steps):
    """Get values at specific steps via nearest interpolation."""
    results = {}
    for s in steps:
        exact = df[df['step'] == s]
        if len(exact) > 0:
            results[s] = float(exact['value'].iloc[-1])
        else:
            # Find nearest
            idx = (df['step'] - s).abs().idxmin()
            results[s] = float(df.loc[idx, 'value'])
    return results


def main():
    print("=" * 70)
    print("Stage B1: Selective vs Full Training Baseline Analysis")
    print("=" * 70)

    # ── Load Data ──
    print("\nLoading CSV files...")
    full_train = load_csv(FULL_TRAIN_CSV, "full/train")
    full_cv = load_csv(FULL_CV_CSV, "full/cv")
    sel_train = load_csv(SEL_TRAIN_CSV, "selective/train")
    sel_cv = load_csv(SEL_CV_CSV, "selective/cv")

    # Deduplicate (selective CSV has multiple runs mixed)
    print("\nDeduplicating runs...")
    sel_train = deduplicate_runs(sel_train)
    sel_cv = deduplicate_runs(sel_cv)
    full_train = deduplicate_runs(full_train)
    full_cv = deduplicate_runs(full_cv)

    # ── Timing Analysis ──
    print("\n" + "-" * 50)
    print("Timing Analysis (wall-clock)")
    print("-" * 50)
    full_timing = compute_timing(full_train)
    sel_timing = compute_timing(sel_train)
    print(f"\n  Full training:")
    for k, v in full_timing.items():
        print(f"    {k}: {v}")
    print(f"\n  Selective training:")
    for k, v in sel_timing.items():
        print(f"    {k}: {v}")

    if full_timing.get('seconds_per_step') and sel_timing.get('seconds_per_step'):
        speedup = sel_timing['seconds_per_step'] / full_timing['seconds_per_step']
        print(f"\n  Selective / Full step time ratio: {speedup:.3f}")
        print(f"  -> {'Full is faster' if speedup > 1 else 'Selective is faster'} per step")

    # ── Train Loss Comparison ──
    print("\n" + "-" * 50)
    print("Train Loss Comparison")
    print("-" * 50)

    # Get first/last values
    full_first = full_train['value'].iloc[0]
    full_last = full_train['value'].iloc[-1]
    sel_first = sel_train['value'].iloc[0]
    sel_last = sel_train['value'].iloc[-1]

    print(f"\n  Full:      {full_first:.4f} -> {full_last:.4f}  (drop: {full_first - full_last:.4f})")
    print(f"  Selective: {sel_first:.4f} -> {sel_last:.4f}  (drop: {sel_first - sel_last:.4f})")

    # ── CV Loss Comparison ──
    print("\n" + "-" * 50)
    print("CV Loss Comparison (step-aligned)")
    print("-" * 50)

    cv_steps = sorted(set(full_cv['step'].tolist()) & set(sel_cv['step'].tolist()))
    if not cv_steps:
        cv_steps = sorted(full_cv['step'].unique().tolist())

    full_cv_vals = interpolate_at_steps(full_cv, cv_steps)
    sel_cv_vals = interpolate_at_steps(sel_cv, cv_steps)

    print(f"\n  {'Step':>6} | {'Full CV':>10} | {'Sel CV':>10} | {'Diff (F-S)':>10}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for s in cv_steps:
        diff = full_cv_vals[s] - sel_cv_vals[s]
        print(f"  {s:>6} | {full_cv_vals[s]:>10.4f} | {sel_cv_vals[s]:>10.4f} | {diff:>+10.4f}")

    # Final CV comparison
    final_full_cv = full_cv_vals[cv_steps[-1]]
    final_sel_cv = sel_cv_vals[cv_steps[-1]]
    print(f"\n  Final CV loss:")
    print(f"    Full:      {final_full_cv:.5f}")
    print(f"    Selective: {final_sel_cv:.5f}")
    print(f"    Diff:      {final_full_cv - final_sel_cv:+.5f}")
    print(f"    -> {'Full' if final_full_cv < final_sel_cv else 'Selective'} has lower final CV loss")

    # ── Wall-Clock Efficiency ──
    print("\n" + "-" * 50)
    print("Wall-Clock Efficiency (CV loss per unit time)")
    print("-" * 50)

    if len(full_cv) >= 2 and len(sel_cv) >= 2:
        full_cv_time = full_cv['wall_time'].iloc[-1] - full_cv['wall_time'].iloc[0]
        sel_cv_time = sel_cv['wall_time'].iloc[-1] - sel_cv['wall_time'].iloc[0]
        full_cv_drop = full_cv['value'].iloc[0] - full_cv['value'].iloc[-1]
        sel_cv_drop = sel_cv['value'].iloc[0] - sel_cv['value'].iloc[-1]

        print(f"\n  Full:      CV drop={full_cv_drop:.4f} in {full_cv_time:.1f}s  "
              f"({full_cv_drop/full_cv_time*1000:.4f} loss/ks)")
        print(f"  Selective: CV drop={sel_cv_drop:.4f} in {sel_cv_time:.1f}s  "
              f"({sel_cv_drop/sel_cv_time*1000:.4f} loss/ks)")

    # ── Smoothed Train Loss Comparison ──
    # Use rolling window for smoother comparison
    window = 10
    full_train_smooth = full_train.copy()
    full_train_smooth['smooth'] = full_train_smooth['value'].rolling(window, min_periods=1).mean()
    sel_train_smooth = sel_train.copy()
    sel_train_smooth['smooth'] = sel_train_smooth['value'].rolling(window, min_periods=1).mean()

    # ══════════════════════════════════════════
    # Figures
    # ══════════════════════════════════════════
    print("\nGenerating figures...")

    # Fig 1: Train loss curves (raw + smoothed)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.plot(full_train['step'], full_train['value'], alpha=0.3, color='blue')
    ax.plot(full_train_smooth['step'], full_train_smooth['smooth'],
            color='blue', linewidth=2, label=f'Full (final={full_last:.4f})')
    ax.plot(sel_train['step'], sel_train['value'], alpha=0.3, color='red')
    ax.plot(sel_train_smooth['step'], sel_train_smooth['smooth'],
            color='red', linewidth=2, label=f'Selective (final={sel_last:.4f})')
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Train Loss', fontsize=12)
    ax.set_title('B1: Train Loss Curves', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(list(full_cv_vals.keys()), list(full_cv_vals.values()),
            'b-o', linewidth=2, markersize=8, label=f'Full (final={final_full_cv:.4f})')
    ax.plot(list(sel_cv_vals.keys()), list(sel_cv_vals.values()),
            'r-o', linewidth=2, markersize=8, label=f'Selective (final={final_sel_cv:.4f})')
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('CV Loss', fontsize=12)
    ax.set_title('B1: CV Loss Curves', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Stage B1: Selective vs Full Training Baseline', fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'b1_loss_curves.png', dpi=150)
    plt.close(fig)

    # Fig 2: CV loss bar chart at each checkpoint
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(cv_steps))
    width = 0.35
    ax.bar(x_pos - width/2, [full_cv_vals[s] for s in cv_steps],
           width, label='Full', color='steelblue', edgecolor='black', alpha=0.8)
    ax.bar(x_pos + width/2, [sel_cv_vals[s] for s in cv_steps],
           width, label='Selective', color='coral', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('CV Loss', fontsize=12)
    ax.set_title('B1: CV Loss at Checkpoints', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(s) for s in cv_steps])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, s in enumerate(cv_steps):
        ax.text(i - width/2, full_cv_vals[s] + 0.005, f'{full_cv_vals[s]:.3f}',
                ha='center', fontsize=9)
        ax.text(i + width/2, sel_cv_vals[s] + 0.005, f'{sel_cv_vals[s]:.3f}',
                ha='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'b1_cv_comparison.png', dpi=150)
    plt.close(fig)

    # Fig 3: Wall-clock aligned train loss
    fig, ax = plt.subplots(figsize=(10, 6))
    full_t0 = full_train['wall_time'].iloc[0]
    sel_t0 = sel_train['wall_time'].iloc[0]
    ax.plot(full_train['wall_time'] - full_t0, full_train['value'],
            alpha=0.3, color='blue')
    ax.plot(full_train_smooth['wall_time'] - full_t0, full_train_smooth['smooth'],
            color='blue', linewidth=2, label='Full')
    ax.plot(sel_train['wall_time'] - sel_t0, sel_train['value'],
            alpha=0.3, color='red')
    ax.plot(sel_train_smooth['wall_time'] - sel_t0, sel_train_smooth['smooth'],
            color='red', linewidth=2, label='Selective')
    ax.set_xlabel('Wall-clock Time (seconds)', fontsize=12)
    ax.set_ylabel('Train Loss', fontsize=12)
    ax.set_title('B1: Train Loss vs Wall-Clock Time', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'b1_wallclock_train_loss.png', dpi=150)
    plt.close(fig)

    # Fig 4: Summary comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    metrics = ['Final Train Loss', 'Final CV Loss']
    full_vals = [full_last, final_full_cv]
    sel_vals = [sel_last, final_sel_cv]
    x_pos = np.arange(len(metrics))
    width = 0.3
    bars1 = ax.bar(x_pos - width/2, full_vals, width, label='Full', color='steelblue',
                   edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, sel_vals, width, label='Selective', color='coral',
                   edgecolor='black', alpha=0.8)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('B1: Final Loss Comparison', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.4f}', ha='center', fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.4f}', ha='center', fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'b1_final_comparison.png', dpi=150)
    plt.close(fig)

    print(f"\n[B1] Figures saved to {FIGURES_DIR}/b1_*.png")

    # ── Overall B1 Conclusion ──
    print("\n" + "=" * 70)
    print("STAGE B1 CONCLUSION")
    print("=" * 70)

    sel_train_better = sel_last < full_last
    full_cv_better = final_full_cv < final_sel_cv
    full_faster = (full_timing.get('seconds_per_step', 0) <
                   sel_timing.get('seconds_per_step', float('inf')))

    print(f"""
  Train loss: {'Selective' if sel_train_better else 'Full'} is lower ({sel_last:.4f} vs {full_last:.4f})
  CV loss:    {'Full' if full_cv_better else 'Selective'} is lower ({final_full_cv:.5f} vs {final_sel_cv:.5f})
  Speed:      {'Full' if full_faster else 'Selective'} is faster per step

  Overall: {'Full baseline slightly outperforms selective in generalization (CV loss)'
            if full_cv_better else
            'Selective outperforms full baseline in generalization (CV loss)'},
  but selective shows stronger training-set fitting.
  This is consistent with selective focusing on harder tokens (stronger fitting)
  but not yet translating to better generalization at this scale.
""")

    # Save summary
    summary = {
        'full_training': {
            'final_train_loss': float(full_last),
            'final_cv_loss': float(final_full_cv),
            'timing': full_timing,
        },
        'selective_training': {
            'final_train_loss': float(sel_last),
            'final_cv_loss': float(final_sel_cv),
            'timing': sel_timing,
        },
        'cv_loss_by_step': {
            'steps': cv_steps,
            'full': [full_cv_vals[s] for s in cv_steps],
            'selective': [sel_cv_vals[s] for s in cv_steps],
        },
        'conclusion': {
            'selective_train_better': bool(sel_train_better),
            'full_cv_better': bool(full_cv_better),
            'full_faster_per_step': bool(full_faster),
        },
    }
    summary_path = OUTPUT_DIR / 'stage_b_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == '__main__':
    main()
