"""Batch feature extraction with checkpoint/resume support.

Processes audio features in batches of N samples, saving each batch
to disk. If interrupted, resumes from the last completed batch.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from feature_extraction import FEATURE_COLUMNS, extract_single_audio_features

LOGGER = logging.getLogger("extract_features_batch")


def _worker(item):
    """Module-level worker for ProcessPoolExecutor (must be picklable)."""
    idx, path = item
    p = Path(path)
    if not p.exists():
        feats = {k: np.nan for k in FEATURE_COLUMNS}
        feats["feature_status"] = "missing"
        return idx, feats
    try:
        feats = extract_single_audio_features(str(p))
        feats["feature_status"] = "ok"
        return idx, feats
    except Exception:
        feats = {k: np.nan for k in FEATURE_COLUMNS}
        feats["feature_status"] = "error"
        return idx, feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--n-workers", type=int, default=16)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metadata)
    total = len(df)
    batch_size = args.batch_size
    n_batches = (total + batch_size - 1) // batch_size

    LOGGER.info("Total samples: %d, batch_size: %d, n_batches: %d",
                total, batch_size, n_batches)

    # Find which batches are already done
    done_batches = set()
    for f in outdir.glob("batch_*.parquet"):
        try:
            idx = int(f.stem.split("_")[1])
            done_batches.add(idx)
        except (ValueError, IndexError):
            pass

    LOGGER.info("Already completed batches: %d / %d", len(done_batches), n_batches)

    from concurrent.futures import ProcessPoolExecutor, as_completed

    for batch_idx in range(n_batches):
        if batch_idx in done_batches:
            continue

        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch_file = outdir / ("batch_%04d.parquet" % batch_idx)

        LOGGER.info("Processing batch %d/%d (samples %d-%d)...",
                     batch_idx + 1, n_batches, start, end - 1)
        t0 = time.time()

        # Use multiprocessing within batch
        work_items = []
        for i in range(start, end):
            row = df.iloc[i]
            work_items.append((i, str(Path(args.data_dir) / row["audio_path"])))

        results = {}

        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            futures = {executor.submit(_worker, item): item[0] for item in work_items}
            for future in as_completed(futures):
                idx, feats = future.result()
                results[idx] = feats

        # Build dataframe in order
        rows = [results[i] for i in range(start, end)]
        batch_df = pd.DataFrame(rows)
        batch_df.to_parquet(batch_file, index=False)

        elapsed = time.time() - t0
        LOGGER.info("  Batch %d done in %.1fs (%.2f s/sample). Saved to %s",
                     batch_idx + 1, elapsed, elapsed / (end - start), batch_file)

    # Merge all batches
    LOGGER.info("Merging all batches...")
    all_dfs = []
    for batch_idx in range(n_batches):
        batch_file = outdir / ("batch_%04d.parquet" % batch_idx)
        all_dfs.append(pd.read_parquet(batch_file))

    merged = pd.concat(all_dfs, ignore_index=True)
    merged_file = outdir / "features_cache.parquet"
    merged.to_parquet(merged_file, index=False)
    LOGGER.info("Merged features saved: %s (%d rows)", merged_file, len(merged))

    # Status summary
    status = merged["feature_status"].value_counts()
    LOGGER.info("Feature status: %s", status.to_dict())
    LOGGER.info("DONE")


if __name__ == "__main__":
    main()
