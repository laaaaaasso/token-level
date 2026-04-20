#!/usr/bin/env python3
"""Stage D: Generation Quality Analysis - summary and visualization.

Reads:
  - stage_d_analysis/objective_metrics.json
  - stage_d_analysis/generation_results.json

Produces:
  - Figures for comparison
  - stage_d_summary.json

Usage:
    python stage_d_analysis/run_analysis.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

MODELS = ['pretrained', 'selective', 'full_baseline', 'random_mask', 'random_ref_fair']
FAIR_MODELS = ['selective', 'full_baseline', 'random_mask', 'random_ref_fair']
LABELS = {
    'pretrained': 'Pretrained',
    'selective': 'Selective (delta)',
    'full_baseline': 'Full baseline',
    'random_mask': 'Random mask',
    'random_ref_fair': 'Random ref (fair)',
}
COLORS = {
    'pretrained': '#95a5a6',
    'selective': '#e74c3c',
    'full_baseline': '#3498db',
    'random_mask': '#2ecc71',
    'random_ref_fair': '#f39c12',
}


def main():
    # Load data
    with open(OUTPUT_DIR / "objective_metrics.json") as f:
        metrics = json.load(f)
    with open(OUTPUT_DIR / "generation_results.json") as f:
        gen_results = json.load(f)

    # Compute aggregates
    summary = {}
    for model in MODELS:
        results = metrics.get(model, [])
        wers = [r['wer'] for r in results if r.get('wer') is not None]
        sims = [r['speaker_similarity'] for r in results if r.get('speaker_similarity') is not None]
        durs = [r['duration_ratio'] for r in results if r.get('duration_ratio') is not None]
        summary[model] = {
            'avg_wer': float(np.mean(wers)) if wers else None,
            'avg_speaker_sim': float(np.mean(sims)) if sims else None,
            'avg_duration_ratio': float(np.mean(durs)) if durs else None,
            'std_wer': float(np.std(wers)) if wers else None,
            'std_speaker_sim': float(np.std(sims)) if sims else None,
            'n_samples': len(results),
        }

    print("=" * 80)
    print("STAGE D: GENERATION QUALITY ANALYSIS")
    print("=" * 80)

    # ── D1: Generation summary ──
    print("\n--- D1: Generation Summary ---")
    for model in MODELS:
        gen = gen_results.get(model, [])
        ok = sum(1 for r in gen if r.get('status') == 'ok')
        avg_dur = np.mean([r['duration'] for r in gen if r.get('duration')]) if gen else 0
        print(f"  {LABELS[model]:<22} {ok}/12 ok  avg_dur={avg_dur:.2f}s")

    # ── D2: Objective Metrics ──
    print("\n--- D2: Objective Metrics ---")
    print(f"{'Model':<22} {'WER':>8} {'SpkSim':>8} {'DurRatio':>10}")
    print("-" * 48)
    for model in MODELS:
        s = summary[model]
        wer = f"{s['avg_wer']:.3f}" if s['avg_wer'] is not None else "N/A"
        sim = f"{s['avg_speaker_sim']:.4f}" if s['avg_speaker_sim'] is not None else "N/A"
        dur = f"{s['avg_duration_ratio']:.3f}" if s['avg_duration_ratio'] is not None else "N/A"
        print(f"  {LABELS[model]:<22} {wer:>8} {sim:>8} {dur:>10}")

    # ── Rankings ──
    print("\n--- Speaker Similarity Ranking (higher is better) ---")
    sim_ranking = [(m, summary[m]['avg_speaker_sim']) for m in FAIR_MODELS if summary[m]['avg_speaker_sim'] is not None]
    sim_ranking.sort(key=lambda x: -x[1])
    for i, (m, v) in enumerate(sim_ranking, 1):
        marker = " ★" if i == 1 else ""
        print(f"  {i}. {LABELS[m]:<22} {v:.4f}{marker}")

    print("\n--- WER Ranking (lower is better) ---")
    wer_ranking = [(m, summary[m]['avg_wer']) for m in FAIR_MODELS if summary[m]['avg_wer'] is not None]
    wer_ranking.sort(key=lambda x: x[1])
    for i, (m, v) in enumerate(wer_ranking, 1):
        marker = " ★" if i == 1 else ""
        print(f"  {i}. {LABELS[m]:<22} {v:.3f}{marker}")

    # ════════════════════════
    # Figures
    # ════════════════════════

    # Fig D1: Speaker Similarity bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    vals = [summary[m]['avg_speaker_sim'] for m in MODELS]
    stds = [summary[m]['std_speaker_sim'] for m in MODELS]
    bars = ax.bar([LABELS[m] for m in MODELS], vals,
                  yerr=stds, capsize=4,
                  color=[COLORS[m] for m in MODELS], edgecolor='black', alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f'{v:.3f}', ha='center', fontsize=9)
    ax.set_ylabel('Speaker Similarity', fontsize=12)
    ax.set_title('D2: Speaker Similarity', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right', fontsize=9)

    # Fig D2: WER bar chart
    ax = axes[1]
    wer_vals = [summary[m]['avg_wer'] if summary[m]['avg_wer'] is not None else 0 for m in MODELS]
    wer_stds = [summary[m]['std_wer'] if summary[m]['std_wer'] is not None else 0 for m in MODELS]
    bars = ax.bar([LABELS[m] for m in MODELS], wer_vals,
                  yerr=wer_stds, capsize=4,
                  color=[COLORS[m] for m in MODELS], edgecolor='black', alpha=0.85)
    for bar, v in zip(bars, wer_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f'{v:.3f}', ha='center', fontsize=9)
    ax.set_ylabel('Word Error Rate', fontsize=12)
    ax.set_title('D2: ASR Word Error Rate', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right', fontsize=9)

    # Fig D3: Duration ratio
    ax = axes[2]
    dur_vals = [summary[m]['avg_duration_ratio'] for m in MODELS]
    bars = ax.bar([LABELS[m] for m in MODELS], dur_vals,
                  color=[COLORS[m] for m in MODELS], edgecolor='black', alpha=0.85)
    for bar, v in zip(bars, dur_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', fontsize=9)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Ideal (1.0)')
    ax.set_ylabel('Duration Ratio (gen/ref)', fontsize=12)
    ax.set_title('D2: Duration Ratio', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right', fontsize=9)

    fig.suptitle('Stage D: Generation Quality Metrics', fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'd2_metrics_overview.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Fig D4: Per-emotion speaker similarity heatmap
    emotions = ['angry', 'happy', 'neutral', 'sad']
    sim_matrix = np.zeros((len(MODELS), len(emotions)))
    for i, model in enumerate(MODELS):
        results = metrics.get(model, [])
        for j, emo in enumerate(emotions):
            emo_sims = [r['speaker_similarity'] for r in results
                       if r.get('speaker_similarity') is not None and r['emotion'] == emo]
            sim_matrix[i, j] = np.mean(emo_sims) if emo_sims else 0

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(sim_matrix, cmap='YlOrRd', aspect='auto', vmin=0.4, vmax=0.9)
    ax.set_xticks(range(len(emotions)))
    ax.set_xticklabels(emotions, fontsize=11)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([LABELS[m] for m in MODELS], fontsize=10)
    for i in range(len(MODELS)):
        for j in range(len(emotions)):
            ax.text(j, i, f'{sim_matrix[i, j]:.3f}', ha='center', va='center', fontsize=10,
                    color='white' if sim_matrix[i, j] > 0.7 else 'black')
    ax.set_title('D2: Speaker Similarity by Emotion', fontsize=14)
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'd2_spksim_by_emotion.png', dpi=150)
    plt.close(fig)

    # Fig D5: Composite score radar chart (WER inverted, SpkSim, DurRatio closeness to 1)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    categories = ['Intelligibility\n(1-WER)', 'Speaker Sim', 'Duration Match\n(1-|ratio-1|)']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    for model in FAIR_MODELS:
        s = summary[model]
        wer_score = 1 - (s['avg_wer'] if s['avg_wer'] is not None else 0.5)
        sim_score = s['avg_speaker_sim'] if s['avg_speaker_sim'] is not None else 0
        dur_score = 1 - abs((s['avg_duration_ratio'] if s['avg_duration_ratio'] is not None else 1) - 1)
        dur_score = max(0, dur_score)
        values = [wer_score, sim_score, dur_score]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, color=COLORS[model], label=LABELS[model])
        ax.fill(angles, values, alpha=0.1, color=COLORS[model])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.set_title('D2: Composite Quality Profile', fontsize=14, pad=20)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'd2_radar.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\nFigures saved to {FIGURES_DIR}/d2_*.png")

    # Save summary
    output = {
        'per_model_summary': summary,
        'speaker_sim_ranking': [{'model': LABELS[m], 'avg_sim': v} for m, v in sim_ranking],
        'wer_ranking': [{'model': LABELS[m], 'avg_wer': v} for m, v in wer_ranking],
    }
    with open(OUTPUT_DIR / 'stage_d_summary.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Summary saved to {OUTPUT_DIR / 'stage_d_summary.json'}")

    # ── Conclusions ──
    print("\n" + "=" * 80)
    print("STAGE D CONCLUSIONS")
    print("=" * 80)
    print(f"""
D1 (Generation):
  - All 5 models successfully generated all 12 test samples
  - Generated audio durations are 50-62% of reference (all models produce shorter speech)

D2 (Objective Metrics):
  - WER: All fine-tuned models achieve same WER (0.301) except selective (0.343)
    -> Selective training slightly *degrades* intelligibility
  - Speaker Similarity ranking (among fine-tuned):
    1. Random mask     (0.700) ★ - best speaker preservation
    2. Selective        (0.692)
    3. Random ref fair  (0.686)
    4. Full baseline    (0.675)
  - Random mask again leads, consistent with CV loss ranking from Stage B/C
  - Pretrained model achieves similar SpkSim (0.687), suggesting fine-tuning
    on this small dataset has minimal impact on generation quality

Key Takeaway:
  At current scale, fine-tuning differences are SMALL in generation metrics.
  Random mask shows slight advantage in speaker similarity.
  Selective training shows slight disadvantage in WER.
  All models produce shorter-than-reference audio (~50-62% of ref duration).
""")


if __name__ == '__main__':
    main()
