"""World-model fitting and detector training."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import (
    RecordingDataset,
    balanced_sampler,
    collate_eval,
    collate_train,
    length_preserving_process,
)
from metrics import eer
from model import COW


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size, shuffle, sampler, workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_train if dataset.processed_view else collate_eval,
    )


def train_world(model: COW, records, args, device) -> None:
    dataset = RecordingDataset(
        records,
        sample_rate=args.sample_rate,
        segment_time=args.segment_time,
        train=True,
        processed_view=False,
    )
    loader = make_loader(dataset, args.batch_size, True, None, args.num_workers)
    optim = torch.optim.AdamW(model.world.parameters(), lr=args.world_lr, weight_decay=args.weight_decay)
    model.train()
    processors = ("identity", "gain", "noise", "lowpass")
    for epoch in range(1, args.world_epochs + 1):
        total = 0.0
        steps = 0
        for wave, _label, _path in loader:
            wave = wave.to(device)
            kind = random.choice(processors)
            view = torch.stack([length_preserving_process(w, kind) for w in wave.cpu()]).to(device)
            optim.zero_grad(set_to_none=True)
            loss = model.world_nll(wave) + model.world_nll(view)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.world.parameters(), 5.0)
            optim.step()
            total += float(loss.detach())
            steps += 1
            if args.debug and steps >= 1:
                break
        print(f"world epoch {epoch}: nll {total / max(steps, 1):.4f}")
        if args.debug:
            break
    model.freeze_world_()


def train_detector(model: COW, records, eval_records, args, device) -> None:
    model.freeze_world_()
    dataset = RecordingDataset(
        records,
        sample_rate=args.sample_rate,
        segment_time=args.segment_time,
        train=True,
        processed_view=True,
    )
    loader = make_loader(
        dataset,
        args.batch_size,
        False,
        balanced_sampler(records),
        args.num_workers,
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        model.freeze_world_()
        totals = {"fuse": 0.0, "dyn": 0.0, "ac": 0.0, "con": 0.0, "saturated": 0.0}
        steps = 0
        for wave, paired, label, matched, _path in loader:
            wave = wave.to(device)
            paired = paired.to(device)
            label = label.to(device)
            matched = matched.to(device)
            optim.zero_grad(set_to_none=True)
            out = model.detector_loss(wave, paired, label, matched)
            loss = (
                out["fuse"]
                + args.lambda_dyn * out["dyn"]
                + args.lambda_ac * out["ac"]
                + args.lambda_con * out["con"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optim.step()
            model.project_heads_()
            for key in totals:
                totals[key] += float(out[key].detach())
            steps += 1
            if args.debug and steps >= 1:
                break
        msg = " ".join(f"{k} {v / max(steps, 1):.4f}" for k, v in totals.items())
        print(f"detector epoch {epoch}: {msg}")
        if eval_records and not args.debug:
            metrics = evaluate(model, eval_records, args, device)
            print(
                f"eval: EER {metrics['eer'] * 100:.2f}%  "
                f"n {metrics['n']}"
            )
        if args.debug:
            break


@torch.no_grad()
def evaluate(model: COW, records, args, device) -> dict:
    if args.max_length is not None and len(records) > args.max_length:
        rng = random.Random(args.seed)
        records = rng.sample(list(records), args.max_length)
    dataset = RecordingDataset(
        records,
        sample_rate=args.sample_rate,
        segment_time=args.segment_time,
        train=False,
        processed_view=False,
    )
    loader = make_loader(dataset, args.eval_batch_size, False, None, args.num_workers)
    model.eval()
    scores, labels = [], []
    for wave, label, _path in loader:
        pred = model.scores(wave.to(device))["fused"]
        scores.append(pred.cpu())
        labels.append(label.cpu())
    score = torch.cat(scores)
    label = torch.cat(labels)
    return {"eer": eer(score, label), "n": int(label.numel())}
