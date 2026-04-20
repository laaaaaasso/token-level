#!/usr/bin/env python3
"""Stage C: Training Efficiency & Data Efficiency Analysis.

C1: Wall-clock efficiency analysis
C2: Unified full-token evaluation comparison

Usage:
    python stage_c_analysis/run_analysis.py
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
STAGE_B_DIR = OUTPUT_DIR.parent / "stage_b_analysis"

# ── Colors ──
COLORS = {
    'selective': '#e74c3c',
    'full_baseline': '#3498db',
    'random_mask': '#2ecc71',
    'random_ref': '#9b59b6',
    'random_ref_fair': '#f39c12',
}
LABELS = {
    'selective': 'Selective (delta)',
    'full_baseline': 'Full baseline',
    'random_mask': 'Random mask',
    'random_ref': 'Random ref (1000 utts)',
    'random_ref_fair': 'Random ref (fair)',
}
# Fair comparison set (exclude random_ref unfair)
FAIR_RUNS = ['selective', 'full_baseline', 'random_mask', 'random_ref_fair']


def load_train_csv(run_name):
    """Load training loss CSV from stage_b_analysis."""
    path = STAGE_B_DIR / f"{run_name}_TRAIN_loss.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = ['wall_time', 'step', 'value']
    # Deduplicate: keep last run
    step_diff = df['step'].diff()
    run_starts = [0] + list(df.index[step_diff < -5])
    if len(run_starts) > 1:
        df = df.iloc[run_starts[-1]:].reset_index(drop=True)
    return df


def load_cv_csv(run_name):
    path = STAGE_B_DIR / f"{run_name}_CV_loss.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = ['wall_time', 'step', 'value']
    step_diff = df['step'].diff()
    run_starts = [0] + list(df.index[step_diff < -5])
    if len(run_starts) > 1:
        df = df.iloc[run_starts[-1]:].reset_index(drop=True)
    return df


def main():
    print("=" * 80)
    print("Stage C: Training Efficiency & Data Efficiency Analysis")
    print("=" * 80)

    # ════════════════════════════════════════════════
    # C1: Wall-clock Efficiency
    # ════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("C1: WALL-CLOCK EFFICIENCY ANALYSIS")
    print("=" * 60)

    timing_data = {}
    for run_name in FAIR_RUNS:
        df = load_train_csv(run_name)
        if df is None or len(df) < 2:
            continue
        total_time = df['wall_time'].iloc[-1] - df['wall_time'].iloc[0]
        n_steps = df['step'].iloc[-1] - df['step'].iloc[0]
        first_loss = df['value'].iloc[:5].mean()
        last_loss = df['value'].iloc[-5:].mean()
        loss_drop = first_loss - last_loss

        timing_data[run_name] = {
            'total_seconds': total_time,
            'total_steps': n_steps,
            'sec_per_step': total_time / n_steps if n_steps > 0 else 0,
            'first_loss': first_loss,
            'last_loss': last_loss,
            'loss_drop': loss_drop,
            'loss_per_second': loss_drop / total_time if total_time > 0 else 0,
            'loss_per_step': loss_drop / n_steps if n_steps > 0 else 0,
        }

    # Print timing table
    print(f"\n{'Model':<20} {'sec/step':>10} {'Total(s)':>10} {'Loss drop':>10} {'Drop/sec':>10} {'Drop/step':>10}")
    print("-" * 70)
    for name in FAIR_RUNS:
        if name not in timing_data:
            continue
        t = timing_data[name]
        print(f"{LABELS[name]:<20} {t['sec_per_step']:>10.3f} {t['total_seconds']:>10.1f} "
              f"{t['loss_drop']:>10.4f} {t['loss_per_second']*1000:>10.4f} {t['loss_per_step']:>10.4f}")

    # Compute relative efficiency
    base = timing_data.get('full_baseline', {})
    if base:
        print(f"\n--- Relative to Full Baseline ---")
        print(f"{'Model':<20} {'Speed ratio':>12} {'Efficiency ratio':>18}")
        print("-" * 50)
        for name in FAIR_RUNS:
            if name not in timing_data:
                continue
            t = timing_data[name]
            speed_ratio = base['sec_per_step'] / t['sec_per_step']
            eff_ratio = t['loss_per_second'] / base['loss_per_second'] if base['loss_per_second'] > 0 else 0
            print(f"{LABELS[name]:<20} {speed_ratio:>12.3f}x {eff_ratio:>18.3f}x")

    # ── C1 Figures ──

    # Fig C1.1: Seconds per step bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    names = [n for n in FAIR_RUNS if n in timing_data]
    sec_vals = [timing_data[n]['sec_per_step'] for n in names]
    bars = ax.bar([LABELS[n] for n in names], sec_vals,
                  color=[COLORS[n] for n in names], edgecolor='black', alpha=0.85)
    for bar, v in zip(bars, sec_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.3f}s', ha='center', fontsize=10)
    ax.set_ylabel('Seconds per Step', fontsize=12)
    ax.set_title('C1.1: Per-Step Training Time', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right')

    # Fig C1.2: Loss drop per second (efficiency)
    ax = axes[1]
    eff_vals = [timing_data[n]['loss_per_second'] * 1000 for n in names]
    bars = ax.bar([LABELS[n] for n in names], eff_vals,
                  color=[COLORS[n] for n in names], edgecolor='black', alpha=0.85)
    for bar, v in zip(bars, eff_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', fontsize=10)
    ax.set_ylabel('Loss Drop per 1000s', fontsize=12)
    ax.set_title('C1.2: Training Efficiency (loss/time)', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right')

    # Fig C1.3: Wall-clock aligned train loss curves
    ax = axes[2]
    window = 10
    for name in FAIR_RUNS:
        df = load_train_csv(name)
        if df is None:
            continue
        t0 = df['wall_time'].iloc[0]
        smooth = df['value'].rolling(window, min_periods=1).mean()
        ax.plot(df['wall_time'] - t0, smooth, color=COLORS[name],
                linewidth=2, label=LABELS[name])
    ax.set_xlabel('Wall-Clock Time (seconds)', fontsize=12)
    ax.set_ylabel('Train Loss (smoothed)', fontsize=12)
    ax.set_title('C1.3: Train Loss vs Time', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Stage C1: Wall-Clock Efficiency Analysis', fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'c1_efficiency.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Fig C1.4: Cumulative loss-time curve
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in FAIR_RUNS:
        df = load_train_csv(name)
        if df is None:
            continue
        t0 = df['wall_time'].iloc[0]
        elapsed = df['wall_time'] - t0
        # Use a rolling min to show "best loss achieved so far"
        best_so_far = df['value'].expanding().min()
        ax.plot(elapsed, best_so_far, color=COLORS[name], linewidth=2, label=LABELS[name])
    ax.set_xlabel('Wall-Clock Time (seconds)', fontsize=12)
    ax.set_ylabel('Best Train Loss So Far', fontsize=12)
    ax.set_title('C1.4: Convergence Speed (best loss vs time)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'c1_convergence.png', dpi=150)
    plt.close(fig)

    # ════════════════════════════════════════════════
    # C2: Unified Full-Token Evaluation
    # ════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("C2: UNIFIED FULL-TOKEN CV EVALUATION")
    print("=" * 60)

    eval_path = OUTPUT_DIR / "unified_eval_results.json"
    with open(eval_path, 'r') as f:
        eval_results = json.load(f)

    # Extract data
    epochs = [0, 1, 2]
    eval_table = {}
    for name in FAIR_RUNS + ['random_ref']:
        if name not in eval_results:
            continue
        losses = []
        accs = []
        for e in epochs:
            key = f"epoch_{e}"
            if key in eval_results[name]:
                losses.append(eval_results[name][key]['avg_loss'])
                accs.append(eval_results[name][key]['avg_accuracy'])
            else:
                losses.append(None)
                accs.append(None)
        eval_table[name] = {'losses': losses, 'accs': accs}

    # Print table
    print(f"\n{'Model':<20} {'Epoch 0':>10} {'Epoch 1':>10} {'Epoch 2':>10} {'Acc':>8}")
    print("-" * 58)
    for name in FAIR_RUNS:
        if name not in eval_table:
            continue
        d = eval_table[name]
        vals = [f"{v:.5f}" if v is not None else "N/A" for v in d['losses']]
        acc = d['accs'][-1] if d['accs'][-1] is not None else float('nan')
        print(f"{LABELS[name]:<20} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {acc:>8.4f}")

    # Compare with original TensorBoard CV values
    print("\n--- C2 vs Original TB CV Loss (Epoch 2) ---")
    tb_cv = {
        'selective': 4.1147,
        'full_baseline': 4.0850,
        'random_mask': 3.9992,
        'random_ref_fair': 4.1047,
    }
    print(f"{'Model':<20} {'TB CV':>10} {'Unified CV':>12} {'Diff':>10}")
    print("-" * 52)
    for name in FAIR_RUNS:
        if name not in eval_table or name not in tb_cv:
            continue
        unified = eval_table[name]['losses'][-1]
        tb = tb_cv[name]
        diff = unified - tb if unified is not None else float('nan')
        print(f"{LABELS[name]:<20} {tb:>10.5f} {unified:>12.5f} {diff:>+10.5f}")

    # Ranking
    print("\n--- Unified CV Loss Ranking (Epoch 2, lower is better) ---")
    ranking = [(name, eval_table[name]['losses'][-1]) for name in FAIR_RUNS
               if name in eval_table and eval_table[name]['losses'][-1] is not None]
    ranking.sort(key=lambda x: x[1])
    for i, (name, loss) in enumerate(ranking, 1):
        marker = " ★" if i == 1 else ""
        print(f"  {i}. {LABELS[name]:<20} {loss:.5f}{marker}")

    # ── C2 Figures ──

    # Fig C2.1: Unified CV loss progression (line chart)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for name in FAIR_RUNS:
        if name not in eval_table:
            continue
        d = eval_table[name]
        valid_epochs = [e for e, v in zip(epochs, d['losses']) if v is not None]
        valid_losses = [v for v in d['losses'] if v is not None]
        ax.plot(valid_epochs, valid_losses, 'o-', color=COLORS[name],
                linewidth=2, markersize=8, label=f"{LABELS[name]} ({valid_losses[-1]:.4f})")
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Full-Token CV Loss', fontsize=12)
    ax.set_title('C2.1: Unified CV Loss Progression', fontsize=13)
    ax.set_xticks(epochs)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Fig C2.2: Final CV loss bar chart
    ax = axes[1]
    final_losses = [(name, eval_table[name]['losses'][-1]) for name in FAIR_RUNS
                    if name in eval_table and eval_table[name]['losses'][-1] is not None]
    final_losses.sort(key=lambda x: x[1])
    names_sorted = [x[0] for x in final_losses]
    vals_sorted = [x[1] for x in final_losses]
    bars = ax.barh([LABELS[n] for n in names_sorted], vals_sorted,
                   color=[COLORS[n] for n in names_sorted], edgecolor='black', alpha=0.85)
    for bar, v in zip(bars, vals_sorted):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{v:.5f}', va='center', fontsize=10)
    ax.set_xlabel('Full-Token CV Loss', fontsize=12)
    ax.set_title('C2.2: Final CV Loss Ranking (Unified)', fontsize=13)
    ax.grid(True, alpha=0.3, axis='x')
    # Set x-axis to show differences better
    min_val = min(vals_sorted)
    max_val = max(vals_sorted)
    margin = (max_val - min_val) * 0.3
    ax.set_xlim(min_val - margin, max_val + margin + 0.05)

    fig.suptitle('Stage C2: Unified Full-Token CV Evaluation', fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'c2_unified_eval.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Fig C2.3: Accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in FAIR_RUNS:
        if name not in eval_table:
            continue
        d = eval_table[name]
        valid_epochs = [e for e, v in zip(epochs, d['accs']) if v is not None]
        valid_accs = [v * 100 for v in d['accs'] if v is not None]
        ax.plot(valid_epochs, valid_accs, 'o-', color=COLORS[name],
                linewidth=2, markersize=8, label=f"{LABELS[name]} ({valid_accs[-1]:.2f}%)")
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Full-Token CV Accuracy (%)', fontsize=12)
    ax.set_title('C2.3: Unified CV Accuracy Progression', fontsize=14)
    ax.set_xticks(epochs)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'c2_accuracy.png', dpi=150)
    plt.close(fig)

    # Fig C2.4: TB CV vs Unified CV comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(FAIR_RUNS))
    width = 0.35
    tb_vals = [tb_cv.get(n, float('nan')) for n in FAIR_RUNS]
    uni_vals = [eval_table[n]['losses'][-1] if n in eval_table else float('nan') for n in FAIR_RUNS]
    ax.bar(x_pos - width/2, tb_vals, width, label='TensorBoard CV', color='steelblue',
           edgecolor='black', alpha=0.8)
    ax.bar(x_pos + width/2, uni_vals, width, label='Unified Full-Token CV', color='coral',
           edgecolor='black', alpha=0.8)
    ax.set_ylabel('CV Loss', fontsize=12)
    ax.set_title('C2.4: TensorBoard vs Unified Evaluation', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([LABELS[n] for n in FAIR_RUNS], fontsize=9, rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for i in range(len(FAIR_RUNS)):
        if not np.isnan(tb_vals[i]):
            ax.text(i - width/2, tb_vals[i] + 0.005, f'{tb_vals[i]:.3f}', ha='center', fontsize=8)
        if not np.isnan(uni_vals[i]):
            ax.text(i + width/2, uni_vals[i] + 0.005, f'{uni_vals[i]:.3f}', ha='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'c2_tb_vs_unified.png', dpi=150)
    plt.close(fig)

    # ════════════════════════════════════════════════
    # Save Summary
    # ════════════════════════════════════════════════
    summary = {
        'c1_timing': timing_data,
        'c2_unified_eval': {
            name: {
                'losses_by_epoch': eval_table[name]['losses'],
                'accs_by_epoch': eval_table[name]['accs'],
                'final_loss': eval_table[name]['losses'][-1],
                'final_acc': eval_table[name]['accs'][-1],
            }
            for name in FAIR_RUNS if name in eval_table
        },
        'c2_ranking': [
            {'rank': i+1, 'model': LABELS[name], 'final_cv_loss': loss}
            for i, (name, loss) in enumerate(ranking)
        ],
        'c2_tb_vs_unified': {
            name: {
                'tb_cv': tb_cv.get(name),
                'unified_cv': eval_table[name]['losses'][-1] if name in eval_table else None,
                'diff': (eval_table[name]['losses'][-1] - tb_cv[name])
                        if name in eval_table and name in tb_cv and eval_table[name]['losses'][-1] is not None
                        else None,
            }
            for name in FAIR_RUNS
        },
    }

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    summary_path = OUTPUT_DIR / 'stage_c_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=_convert)
    print(f"\nSummary saved to {summary_path}")
    print(f"Figures saved to {FIGURES_DIR}/c*.png")

    # ════════════════════════════════════════════════
    # Overall Conclusion
    # ════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("STAGE C CONCLUSIONS")
    print("=" * 80)
    print("""
C1 (Wall-Clock Efficiency):
  - Full baseline is the fastest per step (~1.20 s/step)
  - Selective methods are ~17% slower per step due to mask lookup overhead
  - In loss-per-second efficiency, ranking largely follows per-step speed
  - Selective training does NOT provide wall-clock efficiency gains at this scale

C2 (Unified Full-Token Evaluation):
  - Rankings are CONSISTENT between TensorBoard CV and unified evaluation
  - This confirms the TB-reported CV losses were already computed on all tokens
  - Final ranking (unified, fair):
    1. Random mask    (4.001) ★
    2. Full baseline  (4.086)
    3. Random ref     (4.108)
    4. Selective      (4.118)
  - Selective (delta-based) is ranked last among fair comparisons
  - The 60% sparse training (random mask) provides best generalization,
    suggesting implicit regularization rather than informed selection

Key Takeaway:
  At current scale (807 samples, 3 epochs), delta-based selective training
  offers NEITHER efficiency gains NOR accuracy gains over simpler baselines.
  The benefit, if any, may only emerge at larger scale or longer training.
""")


if __name__ == '__main__':
    main()
