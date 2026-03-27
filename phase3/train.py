#!/usr/bin/env python3
"""Phase 3: Selective SLM training with Phase 2 token masks.

Trains CosyVoice2's text-speech LM using selective backprop:
only top-k% speech tokens (selected by Phase 2 excess loss) contribute to the
training loss. Everything else (optimizer, scheduler, DDP, checkpointing)
reuses CosyVoice2's standard infrastructure.

Usage:
    torchrun --standalone --nproc_per_node=1 -m phase3.train \
        --config phase1_outputs/phase1_cosyvoice2.yaml \
        --train_data phase2_outputs/data/train.data.list \
        --cv_data phase1_outputs/data/cv.data.list \
        --qwen_pretrain_path pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN \
        --onnx_path pretrained_models/CosyVoice2-0.5B \
        --checkpoint phase1_outputs/rm/rm_frozen.pt \
        --mask_path phase2_outputs/token_scores.pt \
        --model_dir phase3_outputs/checkpoints \
        --tensorboard_dir phase3_outputs/tensorboard
"""
from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
import types
from copy import deepcopy

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence, unpad_sequence

logging.getLogger('matplotlib').setLevel(logging.WARNING)


def get_args():
    parser = argparse.ArgumentParser(description='Phase 3: selective SLM training')
    parser.add_argument('--train_engine', default='torch_ddp',
                        choices=['torch_ddp', 'deepspeed'])
    parser.add_argument('--config', required=True, help='CosyVoice2 YAML config')
    parser.add_argument('--train_data', required=True, help='train data.list')
    parser.add_argument('--cv_data', required=True, help='cv data.list')
    parser.add_argument('--qwen_pretrain_path', required=True)
    parser.add_argument('--onnx_path', required=True)
    parser.add_argument('--checkpoint', default=None, help='init checkpoint (.pt)')
    parser.add_argument('--mask_path', required=True,
                        help='Phase 2 token_scores.pt with per-sample masks')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--tensorboard_dir', default='tensorboard')
    parser.add_argument('--ddp.dist_backend', dest='dist_backend',
                        default='nccl', choices=['nccl', 'gloo'])
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--prefetch', default=100, type=int)
    parser.add_argument('--pin_memory', action='store_true', default=False)
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--timeout', default=60, type=int)
    # deepspeed (required by init but we default to torch_ddp)
    parser.add_argument('--deepspeed_config', default=None)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args


def setup_paths():
    """Add CosyVoice to sys.path."""
    for candidate in [
        "/data/zhenghao/repos/CosyVoice",
        str(os.path.join(os.path.dirname(__file__), '..', '..', 'CosyVoice')),
    ]:
        if os.path.isdir(candidate):
            for p in [candidate, os.path.join(candidate, 'third_party', 'Matcha-TTS')]:
                if p not in sys.path:
                    sys.path.insert(0, p)
            return candidate
    raise RuntimeError("CosyVoice root not found")


def create_selective_forward(mask_dict):
    """Create a patched forward method that uses selective SLM loss.

    Forces unistream sequence layout to ensure alignment with Phase 2 masks.
    Uses selective loss: only masked speech tokens contribute.
    Falls back to standard (all speech tokens) when mask is not available
    for a sample (e.g. during CV with unseen utts).
    """
    from cosyvoice.utils.common import IGNORE_ID, th_accuracy
    from phase3.selective_loss import compute_selective_slm_loss

    def selective_forward(self, batch, device):
        # 1. Encode text tokens
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        text_token_emb = self.llm.model.model.embed_tokens(text_token)

        # 2. Extract / load speech tokens
        if 'speech_token' not in batch:
            speech_token, speech_token_len = self.speech_token_extractor.inference(
                batch['whisper_feat'], batch['whisper_feat_len'], device)
        else:
            speech_token = batch['speech_token'].to(device)
            speech_token_len = batch['speech_token_len'].to(device)
        speech_token_emb = self.speech_embedding(speech_token)

        # 3. Special tokens
        sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 4. Build unistream lm_input / lm_target (forced, no bistream)
        text_token_emb_list = unpad_sequence(text_token_emb, text_token_len.cpu(), batch_first=True)
        speech_token_list = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        speech_token_emb_list = unpad_sequence(speech_token_emb, speech_token_len.cpu(), batch_first=True)

        lm_targets = []
        lm_inputs = []
        for i in range(text_token.size(0)):
            t = torch.tensor(
                [IGNORE_ID] * (1 + int(text_token_len[i].item()))
                + speech_token_list[i].tolist()
                + [self.eos_token]
            )
            inp = torch.cat([
                sos_emb.squeeze(0),
                text_token_emb_list[i],
                task_id_emb.squeeze(0),
                speech_token_emb_list[i],
            ], dim=0)
            lm_targets.append(t)
            lm_inputs.append(inp)

        lm_input_len = torch.tensor([x.size(0) for x in lm_inputs], dtype=torch.int32)
        lm_input = pad_sequence(lm_inputs, batch_first=True, padding_value=IGNORE_ID)
        lm_target = pad_sequence(lm_targets, batch_first=True, padding_value=IGNORE_ID).to(device)

        # 5. LLM forward
        lm_output, _ = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)

        # 6. Look up Phase 2 masks for this batch
        utts = batch.get('utts', [])
        speech_masks = [mask_dict.get(utt) for utt in utts]

        # 7. Selective loss
        loss, n_selected = compute_selective_slm_loss(logits, lm_target, speech_masks)
        acc = th_accuracy(
            logits.view(-1, self.llm_decoder.out_features),
            lm_target, ignore_label=IGNORE_ID)

        return {'loss': loss, 'acc': acc}

    return selective_forward


def main():
    args = get_args()
    cosyvoice_root = setup_paths()

    os.environ['onnx_path'] = args.onnx_path
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    import deepspeed
    from hyperpyyaml import load_hyperpyyaml
    from cosyvoice.utils.executor import Executor
    from cosyvoice.utils.train_utils import (
        init_distributed,
        init_dataset_and_dataloader,
        init_optimizer_and_scheduler,
        init_summarywriter, save_model,
        wrap_cuda_model, check_modify_and_save_config,
    )
    from phase3.mask_loader import load_token_masks

    # ── Load config ──
    override_dict = {k: None for k in ['flow', 'hift', 'hifigan']}
    override_dict['qwen_pretrain_path'] = args.qwen_pretrain_path
    with open(args.config, 'r') as f:
        configs = load_hyperpyyaml(f, overrides=override_dict)
    configs['train_conf'].update(vars(args))

    # ── Init distributed ──
    init_distributed(args)

    # ── Dataset & dataloader ──
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
        init_dataset_and_dataloader(args, configs, gan=False, dpo=False)

    # ── Save config ──
    configs = check_modify_and_save_config(args, configs)

    # ── Tensorboard ──
    writer = init_summarywriter(args)

    # ── Load model ──
    model = configs['llm']
    # Always start fresh for Phase 3 (checkpoint is only for model weights)
    start_step, start_epoch = 0, -1
    if args.checkpoint and os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        logging.info("Loaded model weights from: %s", args.checkpoint)
    else:
        logging.warning("No checkpoint loaded (path: %s)", args.checkpoint)

    # ── Load Phase 2 masks ──
    mask_dict = load_token_masks(args.mask_path)

    # ── Patch model forward with selective loss ──
    model.forward = types.MethodType(
        create_selective_forward(mask_dict), model)
    logging.info("Patched model.forward with selective SLM loss")

    # ── Wrap with DDP ──
    model = wrap_cuda_model(args, model)

    # ── Optimizer & scheduler ──
    model, optimizer, scheduler, optimizer_d, scheduler_d = \
        init_optimizer_and_scheduler(args, configs, model, gan=False)
    scheduler.set_step(start_step)

    # ── Save init checkpoint ──
    info_dict = deepcopy(configs['train_conf'])
    info_dict['step'] = start_step
    info_dict['epoch'] = start_epoch
    save_model(model, 'init', info_dict)

    # ── Training loop (standard Executor) ──
    executor = Executor(gan=False)
    executor.step = start_step

    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    logging.info("Starting Phase 3 selective training: step=%d epoch=%d",
                 start_step, start_epoch)

    for epoch in range(start_epoch + 1, info_dict['max_epoch']):
        executor.epoch = epoch
        train_dataset.set_epoch(epoch)
        dist.barrier()
        group_join = dist.new_group(
            backend="gloo",
            timeout=datetime.timedelta(seconds=args.timeout))
        executor.train_one_epoc(
            model, optimizer, scheduler,
            train_data_loader, cv_data_loader,
            writer, info_dict, scaler, group_join)
        dist.destroy_process_group(group_join)

    logging.info("Phase 3 training complete.")


if __name__ == '__main__':
    main()
