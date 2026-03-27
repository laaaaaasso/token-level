"""Phase 1 pipeline — kept for backward compatibility.

The actual Phase 1 implementation is split into:
  - phase1/prepare_data.py   : convert D_ref manifest to CosyVoice2 parquet
  - phase1/export_checkpoint.py : export frozen RM checkpoint
  - scripts/phase1_train.sh  : orchestrates the full training pipeline

See scripts/phase1_train.sh for usage.
"""
