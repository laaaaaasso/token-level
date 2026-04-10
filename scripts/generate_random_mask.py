#!/usr/bin/env python3
"""Generate random mask token_scores.pt for B2 baseline.

Creates masks with the same per-sample selection ratio as the delta-based masks,
but with randomly chosen token positions instead of delta-ranked positions.
"""
import io
import sys
import torch
import numpy as np
from pathlib import Path

def main():
    src = Path("phase2_outputs/token_scores.pt")
    dst = Path("phase2_outputs/token_scores_random_mask.pt")
    
    # Load original (handle format issue)
    content = src.read_bytes()
    buf = io.BytesIO(content)
    data = torch.load(buf, map_location='cpu')
    
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    
    new_results = []
    total_tokens = 0
    selected_tokens = 0
    
    for r in data['results']:
        utt = r['utt']
        orig_mask = r['speech_token_mask']
        n = len(orig_mask)
        k = orig_mask.sum().item()  # Keep same number selected
        
        # Generate random mask with same k
        indices = rng.choice(n, size=k, replace=False)
        random_mask = torch.zeros(n, dtype=torch.bool)
        random_mask[indices] = True
        
        new_results.append({
            'utt': utt,
            'speech_token_score': r['speech_token_score'],  # Keep original scores for reference
            'speech_token_mask': random_mask,
            'target_loss': r['target_loss'],
            'ref_loss': r['ref_loss'],
        })
        total_tokens += n
        selected_tokens += k
    
    out = {
        'results': new_results,
        'topk_ratio': data['topk_ratio'],
        'target_ckpt': data['target_ckpt'],
        'ref_ckpt': data['ref_ckpt'],
        'total_tokens': total_tokens,
        'selected_tokens': selected_tokens,
        'actual_ratio': selected_tokens / total_tokens if total_tokens > 0 else 0,
        'mask_type': 'random',
        'random_seed': 42,
    }
    
    torch.save(out, dst)
    print(f"Saved random mask to {dst}")
    print(f"  Samples: {len(new_results)}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Selected tokens: {selected_tokens}")
    print(f"  Actual ratio: {selected_tokens/total_tokens:.4f}")
    
    # Verify randomness: compare with original
    overlap_ratios = []
    for orig, new in zip(data['results'], new_results):
        orig_m = orig['speech_token_mask']
        new_m = new['speech_token_mask']
        overlap = (orig_m & new_m).sum().item()
        total_sel = orig_m.sum().item()
        overlap_ratios.append(overlap / total_sel if total_sel > 0 else 0)
    
    overlap_ratios = np.array(overlap_ratios)
    print(f"\n  Overlap with delta mask: mean={overlap_ratios.mean():.4f}, std={overlap_ratios.std():.4f}")
    print(f"  (Expected ~0.60 if masks are independent)")

if __name__ == '__main__':
    main()
