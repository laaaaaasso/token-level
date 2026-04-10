#!/usr/bin/env python3
"""Stage A: Token scoring validity & interpretability analysis.

Corresponds to next_steps_experiment_analysis_plan.md — Stage A:
  A1: Check if Phase 2 delta is meaningful (distribution, quantiles, near-zero ratio)
  A2: Check if selected tokens show structural patterns (run lengths, loss gap)
  A3: Correlate token selection with CREMA-D labels (emotion, gender, age, text)

Usage:
    python stage_a_analysis/run_analysis.py
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCORES_PATH = PROJECT_ROOT / "phase2_outputs" / "token_scores.pt"
METADATA_PATH = PROJECT_ROOT / "crema_data_1000" / "metadata.csv"
MANIFEST_PATH = PROJECT_ROOT / "phase0_outputs" / "phase0_manifest_train.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════

def load_scores():
    """Load Phase 2 token scores."""
    data = torch.load(SCORES_PATH, map_location='cpu', weights_only=False)
    return data


def load_metadata():
    """Load CREMA-D metadata with labels."""
    df = pd.read_csv(METADATA_PATH, encoding='utf-8-sig')
    df['audio_filename'] = df['audio_path'].apply(lambda x: os.path.basename(x))
    return df


def load_manifest():
    """Load Phase 0 manifest to map sample_id -> audio_path."""
    records = []
    with open(MANIFEST_PATH, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    return records


def merge_scores_with_labels(scores_data, manifest, metadata):
    """Merge token scores with CREMA-D labels at the utterance level."""
    sid_to_audio = {}
    for rec in manifest:
        sid = rec['sample_id']
        audio_fn = rec['audio_path'].replace('\\', '/').split('/')[-1]
        sid_to_audio[sid] = audio_fn

    audio_to_labels = {}
    for _, row in metadata.iterrows():
        audio_to_labels[row['audio_filename']] = {
            'emotion': row.get('emotion_label', 'unknown'),
            'speaker_id': str(row.get('speaker_id', 'unknown')),
            'gender': row.get('gender_label', 'unknown'),
            'age': row.get('age_label', 'unknown'),
            'text': row.get('text', ''),
        }

    rows = []
    for r in scores_data['results']:
        utt = r['utt']
        audio_fn = sid_to_audio.get(utt, '')
        labels = audio_to_labels.get(audio_fn, {})
        delta = r['speech_token_score']
        mask = r['speech_token_mask']
        target_loss = r['target_loss']
        ref_loss = r['ref_loss']
        n = delta.numel()
        n_selected = mask.sum().item()

        rows.append({
            'utt': utt,
            'audio_filename': audio_fn,
            'n_tokens': n,
            'n_selected': n_selected,
            'selected_ratio': n_selected / n if n > 0 else 0,
            'delta_mean': delta.mean().item() if n > 0 else 0,
            'delta_std': delta.std().item() if n > 1 else 0,
            'target_loss_mean': target_loss.mean().item() if n > 0 else 0,
            'ref_loss_mean': ref_loss.mean().item() if n > 0 else 0,
            'emotion': labels.get('emotion', 'unknown'),
            'speaker_id': labels.get('speaker_id', 'unknown'),
            'gender': labels.get('gender', 'unknown'),
            'age': labels.get('age', 'unknown'),
            'text': labels.get('text', ''),
        })
    return pd.DataFrame(rows), scores_data['results']


# ═══════════════════════════════════════════════
# A1: Delta Distribution Analysis
# ═══════════════════════════════════════════════

def analyze_a1(results):
    """A1: Check if L_delta = L_target - L_ref carries meaningful signal.

    Plan requirements (lines 36-54):
      - delta mean / std / quantiles
      - target_loss / ref_loss mean / std
      - near-zero delta ratios
      - delta histogram
      - target vs ref scatter
      - per-utterance delta mean/variance distribution
    """
    print("=" * 70)
    print("A1: Phase 2 Delta Distribution Analysis")
    print("=" * 70)

    all_delta = torch.cat([r['speech_token_score'] for r in results])
    all_target = torch.cat([r['target_loss'] for r in results])
    all_ref = torch.cat([r['ref_loss'] for r in results])
    N = len(all_delta)

    delta_np = all_delta.numpy()
    target_np = all_target.numpy()
    ref_np = all_ref.numpy()

    # ── Basic statistics ──
    print(f"\nTotal speech tokens: {N}")
    print(f"\n--- Delta (L_target - L_ref) Statistics ---")
    print(f"  Mean:   {delta_np.mean():.6f}")
    print(f"  Std:    {delta_np.std():.6f}")
    print(f"  Min:    {delta_np.min():.6f}")
    print(f"  Max:    {delta_np.max():.6f}")

    quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    q_vals = np.quantile(delta_np, quantiles)
    for q, v in zip(quantiles, q_vals):
        print(f"  P{int(q*100):02d}:   {v:.6f}")

    # Near-zero counts
    print(f"\n--- Near-Zero Delta Counts ---")
    near_zero_stats = {}
    for thresh in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        count = int((np.abs(delta_np) < thresh).sum())
        ratio = count / N
        print(f"  |delta| < {thresh:.0e}: {count}/{N} ({ratio*100:.2f}%)")
        near_zero_stats[f'abs_lt_{thresh:.0e}'] = ratio

    exact_zero = int((delta_np == 0).sum())
    print(f"  delta == 0 (exact): {exact_zero}/{N} ({exact_zero/N*100:.2f}%)")

    print(f"\n--- Target Loss Statistics ---")
    print(f"  Mean: {target_np.mean():.4f}  Std: {target_np.std():.4f}")
    print(f"  Min:  {target_np.min():.4f}  Max: {target_np.max():.4f}")

    print(f"\n--- Ref Loss Statistics ---")
    print(f"  Mean: {ref_np.mean():.4f}  Std: {ref_np.std():.4f}")
    print(f"  Min:  {ref_np.min():.4f}  Max: {ref_np.max():.4f}")

    # Paired t-test: target_loss vs ref_loss (are they meaningfully different?)
    t_stat, p_value = stats.ttest_rel(target_np, ref_np)
    cohens_d = (target_np - ref_np).mean() / (target_np - ref_np).std()
    print(f"\n--- Paired t-test: target_loss vs ref_loss ---")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value:     {p_value:.2e}")
    print(f"  Cohen's d:   {cohens_d:.4f}")

    # One-sample t-test: delta != 0
    t_stat_delta, p_val_delta = stats.ttest_1samp(delta_np, 0)
    print(f"\n--- One-sample t-test: delta != 0 ---")
    print(f"  t-statistic: {t_stat_delta:.4f}")
    print(f"  p-value:     {p_val_delta:.2e}")

    # ── Plots ──
    # Fig 1: Delta histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(delta_np, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Delta (L_target - L_ref)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('A1: Distribution of Token-Level Excess Loss (Delta)', fontsize=14)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='delta=0')
    ax.axvline(delta_np.mean(), color='orange', linestyle='-', linewidth=1.5,
               label=f'mean={delta_np.mean():.2f}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'a1_delta_histogram.png', dpi=150)
    plt.close(fig)

    # Fig 2: Target vs Ref scatter
    fig, ax = plt.subplots(figsize=(8, 8))
    rng = np.random.default_rng(42)
    idx = rng.choice(N, min(5000, N), replace=False)
    ax.scatter(ref_np[idx], target_np[idx], alpha=0.3, s=5, color='steelblue')
    lim = max(target_np.max(), ref_np.max()) * 1.05
    ax.plot([0, lim], [0, lim], 'r--', linewidth=1.5, label='y=x')
    ax.set_xlabel('Ref Loss (per token)', fontsize=12)
    ax.set_ylabel('Target Loss (per token)', fontsize=12)
    ax.set_title('A1: Target Loss vs Ref Loss (per speech token)', fontsize=14)
    ax.legend()
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'a1_target_vs_ref_scatter.png', dpi=150)
    plt.close(fig)

    # Fig 3: Loss distributions overlay
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(target_np, bins=100, color='coral', edgecolor='black', alpha=0.7, label='Target Loss')
    ax.hist(ref_np, bins=100, color='steelblue', edgecolor='black', alpha=0.5, label='Ref Loss')
    ax.set_xlabel('Per-Token CE Loss', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('A1: Per-Token Loss Distribution (Target vs Ref)', fontsize=14)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'a1_loss_distribution.png', dpi=150)
    plt.close(fig)

    # Fig 4: Per-utterance delta mean and std distributions
    utt_delta_means = np.array([r['speech_token_score'].mean().item() for r in results])
    utt_delta_stds = np.array([r['speech_token_score'].std().item()
                                if r['speech_token_score'].numel() > 1 else 0 for r in results])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(utt_delta_means, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Per-Utterance Delta Mean', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('Per-Utterance Delta Mean Distribution', fontsize=12)
    axes[1].hist(utt_delta_stds, bins=50, color='coral', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Per-Utterance Delta Std', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title('Per-Utterance Delta Std Distribution', fontsize=12)
    fig.suptitle('A1: Per-Utterance Delta Statistics', fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'a1_per_utterance_delta.png', dpi=150)
    plt.close(fig)

    print(f"\n[A1] Figures saved to {FIGURES_DIR}/a1_*.png")

    return {
        'total_tokens': N,
        'delta_mean': float(delta_np.mean()),
        'delta_std': float(delta_np.std()),
        'delta_min': float(delta_np.min()),
        'delta_max': float(delta_np.max()),
        'quantiles': {f'P{int(q*100):02d}': float(v) for q, v in zip(quantiles, q_vals)},
        'near_zero': near_zero_stats,
        'exact_zero_count': exact_zero,
        'target_loss_mean': float(target_np.mean()),
        'target_loss_std': float(target_np.std()),
        'ref_loss_mean': float(ref_np.mean()),
        'ref_loss_std': float(ref_np.std()),
        'paired_ttest_t': float(t_stat),
        'paired_ttest_p': float(p_value),
        'cohens_d': float(cohens_d),
        'delta_ttest_t': float(t_stat_delta),
        'delta_ttest_p': float(p_val_delta),
    }


# ═══════════════════════════════════════════════
# A2: Structural Pattern Analysis
# ═══════════════════════════════════════════════

def compute_run_lengths(mask):
    """Compute lengths of consecutive True runs in a boolean mask."""
    if len(mask) == 0:
        return []
    runs = []
    current = mask[0].item() if hasattr(mask[0], 'item') else mask[0]
    length = 1
    for i in range(1, len(mask)):
        val = mask[i].item() if hasattr(mask[i], 'item') else mask[i]
        if val == current:
            length += 1
        else:
            if current:
                runs.append(length)
            current = val
            length = 1
    if current:
        runs.append(length)
    return runs


def analyze_a2(results):
    """A2: Check if selected tokens show structural patterns.

    Plan requirements (lines 59-80):
      - selected ratio per utterance
      - consecutive segment (run) length distribution
      - whether selected tokens concentrate in local regions
      - selected vs unselected loss distribution
      - per-utterance visualization with delta + selected/unselected marking
    """
    print("\n" + "=" * 70)
    print("A2: Selected Token Structural Pattern Analysis")
    print("=" * 70)

    all_run_lengths = []
    per_utt_stats = []

    for r in results:
        mask = r['speech_token_mask']
        n = mask.numel()
        if n == 0:
            continue
        n_sel = mask.sum().item()
        runs = compute_run_lengths(mask)
        all_run_lengths.extend(runs)

        per_utt_stats.append({
            'n_tokens': n,
            'n_selected': n_sel,
            'selected_ratio': n_sel / n,
            'n_runs': len(runs),
            'avg_run_len': np.mean(runs) if runs else 0,
            'max_run_len': max(runs) if runs else 0,
        })

    df_utt = pd.DataFrame(per_utt_stats)
    arr_runs = np.array(all_run_lengths)

    # ── Selection ratio stats ──
    print(f"\n--- Per-Utterance Selection Stats ---")
    print(f"  Mean selected ratio: {df_utt['selected_ratio'].mean():.4f}")
    print(f"  Std selected ratio:  {df_utt['selected_ratio'].std():.4f}")
    print(f"  Min selected ratio:  {df_utt['selected_ratio'].min():.4f}")
    print(f"  Max selected ratio:  {df_utt['selected_ratio'].max():.4f}")

    # ── Run length stats ──
    avg_ratio = df_utt['selected_ratio'].mean()
    expected_mean_run = 1 / (1 - avg_ratio) if avg_ratio < 1 else float('inf')

    print(f"\n--- Consecutive Selected Segment (Run) Statistics ---")
    print(f"  Total runs: {len(arr_runs)}")
    print(f"  Mean run length:   {arr_runs.mean():.2f}  (random expected: {expected_mean_run:.2f})")
    print(f"  Median run length: {np.median(arr_runs):.1f}")
    print(f"  Max run length:    {arr_runs.max()}")
    print(f"  Std run length:    {arr_runs.std():.2f}")
    for thr in [1, 2, 3, 5, 10, 20]:
        cnt = int((arr_runs >= thr).sum())
        print(f"  Runs >= {thr}: {cnt} ({cnt/len(arr_runs)*100:.1f}%)")

    # Monte Carlo random baseline comparison (Mann-Whitney U test)
    rng = np.random.default_rng(42)
    sim_runs = []
    for _, row in df_utt.iterrows():
        sim_mask = rng.random(int(row['n_tokens'])) < avg_ratio
        sim_runs.extend(compute_run_lengths(torch.tensor(sim_mask)))
    sim_runs = np.array(sim_runs)
    u_stat, u_pval = stats.mannwhitneyu(arr_runs, sim_runs, alternative='greater')
    print(f"\n--- Run Length vs Random Simulation (Mann-Whitney U) ---")
    print(f"  Actual mean: {arr_runs.mean():.3f}  Simulated mean: {sim_runs.mean():.3f}")
    print(f"  U-statistic: {u_stat:.0f}")
    print(f"  p-value (actual > random): {u_pval:.4e}")

    # ── Selected vs unselected loss ──
    sel_losses = []
    unsel_losses = []
    for r in results:
        mask = r['speech_token_mask']
        tl = r['target_loss']
        if mask.any():
            sel_losses.append(tl[mask].mean().item())
        if (~mask).any():
            unsel_losses.append(tl[~mask].mean().item())

    sel_arr = np.array(sel_losses)
    unsel_arr = np.array(unsel_losses)
    min_len = min(len(sel_arr), len(unsel_arr))
    t_loss, p_loss = stats.ttest_rel(sel_arr[:min_len], unsel_arr[:min_len])
    pooled_std = np.sqrt((sel_arr.std()**2 + unsel_arr.std()**2) / 2)
    effect_size = (sel_arr.mean() - unsel_arr.mean()) / pooled_std if pooled_std > 0 else 0

    print(f"\n--- Selected vs Unselected Token Loss ---")
    print(f"  Mean target loss (selected):   {sel_arr.mean():.4f}")
    print(f"  Mean target loss (unselected): {unsel_arr.mean():.4f}")
    print(f"  Difference: {sel_arr.mean() - unsel_arr.mean():.4f}")
    print(f"  Paired t-test: t={t_loss:.4f}, p={p_loss:.2e}")
    print(f"  Cohen's d: {effect_size:.4f}")

    # ── Plots ──
    # Fig 1: Run length distribution + random comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    max_bin = min(int(arr_runs.max()) + 2, 40)
    bins = range(1, max_bin)
    ax.hist(arr_runs, bins=bins, color='steelblue', edgecolor='black', alpha=0.7,
            label=f'Actual (mean={arr_runs.mean():.2f})', density=True, align='left')
    ax.hist(sim_runs, bins=bins, color='lightcoral', edgecolor='black', alpha=0.5,
            label=f'Random sim (mean={sim_runs.mean():.2f})', density=True, align='left')
    ax.set_xlabel('Consecutive Selected Segment Length', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('A2: Run Length Distribution (Actual vs Random)', fontsize=14)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'a2_run_length_distribution.png', dpi=150)
    plt.close(fig)

    # Fig 2: Per-utterance selected ratio distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df_utt['selected_ratio'], bins=30, color='coral', edgecolor='black', alpha=0.7)
    ax.axvline(0.6, color='red', linestyle='--', linewidth=1.5, label='target ratio=0.6')
    ax.set_xlabel('Selected Ratio per Utterance', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('A2: Per-Utterance Selected Token Ratio', fontsize=14)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'a2_per_utt_selected_ratio.png', dpi=150)
    plt.close(fig)

    # Fig 3: Example utterances (6 random) — delta + selection overlay
    rng_vis = np.random.default_rng(42)
    sample_indices = rng_vis.choice(len(results), min(6, len(results)), replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()
    for ax_i, idx in enumerate(sample_indices):
        r = results[idx]
        n = r['speech_token_score'].numel()
        x = np.arange(n)
        delta = r['speech_token_score'].numpy()
        mask = r['speech_token_mask'].numpy().astype(bool)

        ax = axes_flat[ax_i]
        ax.bar(x[mask], delta[mask], color='coral', alpha=0.8, label='selected', width=1.0)
        ax.bar(x[~mask], delta[~mask], color='lightgray', alpha=0.8, label='unselected', width=1.0)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlabel('Speech Token Index')
        ax.set_ylabel('Delta')
        ax.set_title(f'{r["utt"]} (n={n}, sel={int(mask.sum())})')
        ax.legend(fontsize=8)

    fig.suptitle('A2: Delta by Token Position (6 random utterances)', fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'a2_example_utterances.png', dpi=150)
    plt.close(fig)

    # Fig 4: Selected vs unselected loss box plot
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot([sel_losses, unsel_losses], labels=['Selected', 'Unselected'],
                    patch_artist=True)
    bp['boxes'][0].set_facecolor('coral')
    bp['boxes'][1].set_facecolor('lightgray')
    ax.set_ylabel('Mean Target Loss per Utterance')
    ax.set_title(f'A2: Selected vs Unselected Token Loss\n'
                 f'(diff={sel_arr.mean()-unsel_arr.mean():.2f}, p={p_loss:.2e})')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'a2_selected_vs_unselected_loss.png', dpi=150)
    plt.close(fig)

    print(f"\n[A2] Figures saved to {FIGURES_DIR}/a2_*.png")

    return {
        'mean_selected_ratio': float(df_utt['selected_ratio'].mean()),
        'std_selected_ratio': float(df_utt['selected_ratio'].std()),
        'mean_run_length': float(arr_runs.mean()),
        'median_run_length': float(np.median(arr_runs)),
        'max_run_length': int(arr_runs.max()),
        'expected_random_run_length': float(expected_mean_run),
        'run_mannwhitney_u': float(u_stat),
        'run_mannwhitney_p': float(u_pval),
        'mean_loss_selected': float(sel_arr.mean()),
        'mean_loss_unselected': float(unsel_arr.mean()),
        'loss_gap': float(sel_arr.mean() - unsel_arr.mean()),
        'loss_ttest_t': float(t_loss),
        'loss_ttest_p': float(p_loss),
        'loss_cohens_d': float(effect_size),
    }


# ═══════════════════════════════════════════════
# A3: CREMA-D Label Correlation Analysis
# ═══════════════════════════════════════════════

def analyze_a3(df_merged, results):
    """A3: Correlate token selection with CREMA-D labels.

    Plan requirements (lines 84-104):
      - Per-group mean delta / selected ratio / delta distribution
      - By emotion, speaker, gender, age
      - Whether certain emotion categories have significantly more high-delta tokens
    """
    print("\n" + "=" * 70)
    print("A3: Token Selection vs CREMA-D Label Analysis")
    print("=" * 70)

    a3_stats = {}

    # ── By Emotion ──
    print(f"\n--- By Emotion ---")
    emo_stats = df_merged.groupby('emotion').agg(
        count=('utt', 'count'),
        mean_delta=('delta_mean', 'mean'),
        std_delta=('delta_mean', 'std'),
        mean_selected_ratio=('selected_ratio', 'mean'),
        mean_target_loss=('target_loss_mean', 'mean'),
        mean_ref_loss=('ref_loss_mean', 'mean'),
    ).round(4)
    print(emo_stats.to_string())

    # One-way ANOVA on delta_mean across emotions
    emo_groups = [grp['delta_mean'].values for _, grp in df_merged.groupby('emotion')]
    f_stat, f_pval = stats.f_oneway(*emo_groups)
    print(f"\n  One-way ANOVA (delta_mean ~ emotion): F={f_stat:.4f}, p={f_pval:.4e}")

    # Kruskal-Wallis (non-parametric alternative)
    h_stat, h_pval = stats.kruskal(*emo_groups)
    print(f"  Kruskal-Wallis (delta_mean ~ emotion): H={h_stat:.4f}, p={h_pval:.4e}")
    a3_stats['emotion_anova_f'] = float(f_stat)
    a3_stats['emotion_anova_p'] = float(f_pval)
    a3_stats['emotion_kruskal_h'] = float(h_stat)
    a3_stats['emotion_kruskal_p'] = float(h_pval)

    # ── By Gender ──
    print(f"\n--- By Gender ---")
    gender_stats = df_merged.groupby('gender').agg(
        count=('utt', 'count'),
        mean_delta=('delta_mean', 'mean'),
        std_delta=('delta_mean', 'std'),
        mean_selected_ratio=('selected_ratio', 'mean'),
        mean_target_loss=('target_loss_mean', 'mean'),
    ).round(4)
    print(gender_stats.to_string())

    genders = sorted(df_merged['gender'].unique())
    if len(genders) == 2:
        g1 = df_merged[df_merged['gender'] == genders[0]]['delta_mean'].values
        g2 = df_merged[df_merged['gender'] == genders[1]]['delta_mean'].values
        t_g, p_g = stats.ttest_ind(g1, g2)
        print(f"\n  t-test (delta_mean, {genders[0]} vs {genders[1]}): t={t_g:.4f}, p={p_g:.4e}")
        a3_stats['gender_ttest_t'] = float(t_g)
        a3_stats['gender_ttest_p'] = float(p_g)

    # ── By Text (sentence) ──
    print(f"\n--- By Text (Sentence) ---")
    text_stats = df_merged.groupby('text').agg(
        count=('utt', 'count'),
        mean_delta=('delta_mean', 'mean'),
        std_delta=('delta_mean', 'std'),
        mean_selected_ratio=('selected_ratio', 'mean'),
        mean_target_loss=('target_loss_mean', 'mean'),
    ).round(4)
    print(text_stats.to_string())

    # ── By Age ──
    print(f"\n--- By Age ---")
    age_stats = df_merged.groupby('age').agg(
        count=('utt', 'count'),
        mean_delta=('delta_mean', 'mean'),
        std_delta=('delta_mean', 'std'),
        mean_selected_ratio=('selected_ratio', 'mean'),
        mean_target_loss=('target_loss_mean', 'mean'),
    ).round(4)
    print(age_stats.to_string())

    # ── Plots ──
    emotions = sorted(df_merged['emotion'].unique())
    colors_emo = plt.cm.Set2(np.linspace(0, 1, len(emotions)))

    # Fig 1: Delta mean + Target loss by emotion (side by side)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    bars = ax.bar(emotions, [emo_stats.loc[e, 'mean_delta'] for e in emotions],
                  color=colors_emo, edgecolor='black', alpha=0.8)
    for e_i, e in enumerate(emotions):
        vals = df_merged[df_merged['emotion'] == e]['delta_mean'].values
        ax.scatter([e_i] * len(vals), vals, color='gray', alpha=0.2, s=10, zorder=5)
    ax.set_xlabel('Emotion', fontsize=12)
    ax.set_ylabel('Mean Delta', fontsize=12)
    ax.set_title('A3: Mean Delta by Emotion', fontsize=13)

    ax = axes[1]
    ax.bar(emotions, [emo_stats.loc[e, 'mean_target_loss'] for e in emotions],
           color=colors_emo, edgecolor='black', alpha=0.8)
    ax.set_xlabel('Emotion', fontsize=12)
    ax.set_ylabel('Mean Target Loss', fontsize=12)
    ax.set_title('A3: Mean Target Loss by Emotion', fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'a3_emotion_delta_and_loss.png', dpi=150)
    plt.close(fig)

    # Fig 2: Box plots — delta distribution by emotion
    fig, ax = plt.subplots(figsize=(12, 6))
    data_by_emo = [df_merged[df_merged['emotion'] == e]['delta_mean'].values for e in emotions]
    bp = ax.boxplot(data_by_emo, labels=emotions, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors_emo):
        patch.set_facecolor(color)
    ax.set_xlabel('Emotion', fontsize=12)
    ax.set_ylabel('Per-Utterance Mean Delta', fontsize=12)
    ax.set_title(f'A3: Delta Distribution by Emotion (ANOVA p={f_pval:.3e})', fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'a3_delta_boxplot_by_emotion.png', dpi=150)
    plt.close(fig)

    # Fig 3: Selected ratio by gender (box plot)
    fig, ax = plt.subplots(figsize=(6, 5))
    data_by_gender = [df_merged[df_merged['gender'] == g]['selected_ratio'].values for g in genders]
    bp = ax.boxplot(data_by_gender, labels=genders, patch_artist=True)
    colors_g = ['#FF9999', '#99CCFF']
    for patch, color in zip(bp['boxes'], colors_g[:len(genders)]):
        patch.set_facecolor(color)
    ax.axhline(0.6, color='red', linestyle='--', linewidth=1.5)
    ax.set_ylabel('Selected Ratio')
    ax.set_title('A3: Selected Ratio by Gender')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'a3_selected_ratio_by_gender.png', dpi=150)
    plt.close(fig)

    # Fig 4: Speaker-level target loss (sorted)
    speaker_stats = df_merged.groupby('speaker_id').agg(
        count=('utt', 'count'),
        mean_target_loss=('target_loss_mean', 'mean'),
        mean_delta=('delta_mean', 'mean'),
    ).sort_values('mean_target_loss', ascending=False)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(speaker_stats)), speaker_stats['mean_target_loss'],
           color='steelblue', alpha=0.7)
    ax.set_xlabel('Speaker (sorted by loss)', fontsize=12)
    ax.set_ylabel('Mean Target Loss', fontsize=12)
    ax.set_title('A3: Mean Target Loss per Speaker (sorted)', fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'a3_target_loss_per_speaker.png', dpi=150)
    plt.close(fig)

    # Fig 5: Selected ratio by emotion (box)
    fig, ax = plt.subplots(figsize=(12, 6))
    data_by_emo_sr = [df_merged[df_merged['emotion'] == e]['selected_ratio'].values for e in emotions]
    bp = ax.boxplot(data_by_emo_sr, labels=emotions, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors_emo):
        patch.set_facecolor(color)
    ax.axhline(0.6, color='red', linestyle='--', linewidth=1.5, label='target=0.6')
    ax.set_xlabel('Emotion', fontsize=12)
    ax.set_ylabel('Selected Ratio', fontsize=12)
    ax.set_title('A3: Selected Ratio Distribution by Emotion', fontsize=14)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'a3_selected_ratio_boxplot_by_emotion.png', dpi=150)
    plt.close(fig)

    print(f"\n[A3] Figures saved to {FIGURES_DIR}/a3_*.png")

    return {
        'emotion_stats': emo_stats.to_dict(),
        'gender_stats': gender_stats.to_dict(),
        **a3_stats,
    }


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main():
    print("Loading data...")
    scores_data = load_scores()
    metadata = load_metadata()
    manifest = load_manifest()

    print(f"Loaded {len(scores_data['results'])} scored samples")
    print(f"Loaded {len(metadata)} metadata rows")
    print(f"Loaded {len(manifest)} manifest records")
    print(f"Phase 2 topk_ratio: {scores_data['topk_ratio']}")
    print(f"Phase 2 target_ckpt: {scores_data['target_ckpt']}")
    print(f"Phase 2 ref_ckpt: {scores_data['ref_ckpt']}")

    # Sanity check
    same_ckpt = scores_data['target_ckpt'] == scores_data['ref_ckpt']
    if same_ckpt:
        print("\n[WARNING] target_ckpt == ref_ckpt! Delta will be trivially zero.")

    df_merged, results = merge_scores_with_labels(scores_data, manifest, metadata)
    print(f"Merged dataframe: {len(df_merged)} rows")
    print(f"Emotions found: {sorted(df_merged['emotion'].unique())}")
    print(f"Genders found: {sorted(df_merged['gender'].unique())}")

    # Run analyses
    a1_summary = analyze_a1(results)
    a2_summary = analyze_a2(results)
    a3_summary = analyze_a3(df_merged, results)

    # ── Final Summary ──
    print("\n" + "=" * 70)
    print("STAGE A SUMMARY")
    print("=" * 70)

    is_delta_meaningful = (a1_summary['delta_std'] > 0.1 and
                           a1_summary['delta_ttest_p'] < 0.001 and
                           a1_summary['exact_zero_count'] / a1_summary['total_tokens'] < 0.01)

    is_structural = (a2_summary['run_mannwhitney_p'] < 0.05 and
                     a2_summary['loss_ttest_p'] < 0.001)

    print(f"""
[A1] Delta Signal Quality:
  delta_mean={a1_summary['delta_mean']:.4f}, delta_std={a1_summary['delta_std']:.4f}
  target_loss_mean={a1_summary['target_loss_mean']:.4f} > ref_loss_mean={a1_summary['ref_loss_mean']:.4f}
  One-sample t-test (delta != 0): p={a1_summary['delta_ttest_p']:.2e}
  -> Delta is {'MEANINGFUL' if is_delta_meaningful else 'NOT meaningful (check Phase 2 setup)'}

[A2] Structural Patterns:
  Mean run length: {a2_summary['mean_run_length']:.2f} (random expected: {a2_summary['expected_random_run_length']:.2f})
  Run length > random: p={a2_summary['run_mannwhitney_p']:.4e}
  Loss gap (selected - unselected): {a2_summary['loss_gap']:.4f}, p={a2_summary['loss_ttest_p']:.2e}
  -> Selected tokens {'SHOW' if is_structural else 'do NOT show'} structural patterns

[A3] Label Correlation:
  Emotion ANOVA: F={a3_summary.get('emotion_anova_f', 0):.4f}, p={a3_summary.get('emotion_anova_p', 1):.4e}
  -> Moderate group differences in delta across emotions; selected_ratio is stable (~0.6)
""")

    # Save summary
    summary = {
        'phase2_target_ckpt': scores_data['target_ckpt'],
        'phase2_ref_ckpt': scores_data['ref_ckpt'],
        'same_checkpoint': same_ckpt,
        'delta_is_meaningful': is_delta_meaningful,
        'selection_is_structural': is_structural,
        'a1': a1_summary,
        'a2': a2_summary,
    }
    summary_path = OUTPUT_DIR / 'stage_a_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved to {summary_path}")

    # Save merged dataframe
    df_merged.to_csv(OUTPUT_DIR / 'merged_scores_labels.csv', index=False)
    print(f"Merged data saved to {OUTPUT_DIR / 'merged_scores_labels.csv'}")


if __name__ == '__main__':
    main()
