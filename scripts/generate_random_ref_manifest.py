#!/usr/bin/env python3
"""Generate a random reference manifest for B3 baseline.

Takes all 1000 CREMA-D samples and randomly selects 849 (same size as D_ref)
without using the statistical scoring criteria.
"""
import json
import random
from pathlib import Path

def main():
    # Read all samples
    all_manifest = Path("phase0_outputs/phase0_manifest_all.jsonl")
    rows = []
    with open(all_manifest) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    
    print(f"Total samples: {len(rows)}")
    
    # Original D_ref size
    ref_manifest = Path("phase0_outputs/phase0_manifest_ref.jsonl")
    ref_rows = []
    with open(ref_manifest) as f:
        for line in f:
            line = line.strip()
            if line:
                ref_rows.append(json.loads(line))
    n_ref = len(ref_rows)
    print(f"Original D_ref size: {n_ref}")
    
    # Random sample (fixed seed for reproducibility)
    rng = random.Random(12345)
    random_ref_rows = rng.sample(rows, n_ref)
    
    # Check overlap with original ref
    orig_ref_ids = {r['sample_id'] for r in ref_rows}
    random_ref_ids = {r['sample_id'] for r in random_ref_rows}
    overlap = orig_ref_ids & random_ref_ids
    print(f"Random ref size: {len(random_ref_rows)}")
    print(f"Overlap with original ref: {len(overlap)} / {n_ref} ({len(overlap)/n_ref*100:.1f}%)")
    
    # Save
    out_path = Path("phase0_outputs/phase0_manifest_random_ref.jsonl")
    with open(out_path, 'w') as f:
        for r in random_ref_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    main()
