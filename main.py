"""COW experiment entry.

Data files use HyperPotter's protocol columns. The model is the conditional
world-model detector: genuine reference fitting, trajectory evidence, capped
consistency, and fixed fusion with an acoustic score.

Examples
--------
Train on ASVspoof 2019 LA, then test In-the-Wild:

    python main.py --stage both ^
        --protocols_trn_path protocols/ASVspoof2019LA.txt ^
        --protocols_wild_path protocols/InTheWild.txt ^
        --comment cow

Evaluate a checkpoint:

    python main.py --dev ^
        --protocols_dev_path protocols/InTheWild.txt ^
        --model_path exp/cow/detector.pt

Smoke test without a corpus:

    python main.py --debug --stage both
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import read_protocol, split_reference, write_debug_protocol
from engine import evaluate, set_seed, train_detector, train_world
from model import COW


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="COW training and evaluation")
    parser.add_argument("--stage", choices=("world", "detector", "both"), default="both")
    parser.add_argument("--dev", action="store_true", help="Evaluate a checkpoint and exit")
    parser.add_argument("--debug", action="store_true", help="Run one step on synthetic audio")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--world_path", type=str, default=None)
    parser.add_argument("--protocols_trn_path", type=str, default="protocols/ASVspoof2019LA.txt")
    parser.add_argument("--protocols_dev_path", type=str, default=None)
    parser.add_argument("--protocols_wild_path", type=str, default="protocols/InTheWild.txt")
    parser.add_argument("--audio_root", type=str, default=None, help="Used only for 5-column ASVspoof protocols")
    parser.add_argument("--comment", type=str, default="cow")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--world_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--world_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=2000, help="Subsample this many eval trials; 0 keeps all")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--segment_time", type=float, default=4.0)
    parser.add_argument("--ref_ratio", type=float, default=0.5)
    parser.add_argument("--interval", type=int, default=1600)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--rff", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--lambda_dyn", type=float, default=1.0)
    parser.add_argument("--lambda_ac", type=float, default=1.0)
    parser.add_argument("--lambda_con", type=float, default=0.1)
    return parser


def build_model(args) -> COW:
    horizons = tuple(h for h in (1, 2, 4) if h <= args.horizon)
    return COW(
        dim=args.dim,
        interval=args.interval,
        history=args.history,
        horizon=args.horizon,
        horizons=horizons,
        rollouts=args.rollouts,
        rff=args.rff,
        alpha=args.alpha,
        tau=args.tau,
    )


def checkpoint(model, args, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, path)
    print(f"saved {path}")


def load_checkpoint(model, path: str, device) -> None:
    blob = torch.load(path, map_location=device)
    model.load_state_dict(blob["model"] if isinstance(blob, dict) and "model" in blob else blob)
    model.freeze_world_()


def main() -> None:
    args = build_parser().parse_args()
    if args.max_length == 0:
        args.max_length = None
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.debug else "cpu")
    if args.debug:
        args.segment_time = 2.0
        args.interval = 800
        args.history = 4
        args.horizon = 2
        args.rollouts = 2
        args.dim = 32
        args.rff = 4
        args.batch_size = 2
        args.world_epochs = 1
        args.num_epochs = 1
        protocol = write_debug_protocol(Path(__file__).resolve().parent)
        args.protocols_trn_path = str(protocol)
        args.protocols_wild_path = None

    model = build_model(args).to(device)
    out_dir = Path("exp") / (args.comment or "cow")

    if args.dev:
        if not args.model_path or not args.protocols_dev_path:
            raise SystemExit("--dev needs --model_path and --protocols_dev_path")
        load_checkpoint(model, args.model_path, device)
        records = read_protocol(
            args.protocols_dev_path,
            splits=["test", "eval", "dev", "-"],
            audio_root=args.audio_root,
        )
        metrics = evaluate(model, records, args, device)
        print(f"EER {metrics['eer'] * 100:.2f}%  n {metrics['n']}  {args.protocols_dev_path}")
        return

    train_records = read_protocol(args.protocols_trn_path, splits=["train"], audio_root=args.audio_root)
    ref_records, det_records = split_reference(train_records, args.ref_ratio, args.seed)
    print(f"reference genuine {len(ref_records)}  detector {len(det_records)}")

    eval_records = []
    if args.protocols_wild_path and Path(args.protocols_wild_path).exists():
        eval_records = read_protocol(
            args.protocols_wild_path,
            splits=["-", "eval", "test", "dev"],
            audio_root=args.audio_root,
        )

    if args.stage in {"detector"} and args.world_path:
        load_checkpoint(model, args.world_path, device)
    if args.stage in {"world", "both"}:
        train_world(model, ref_records, args, device)
        checkpoint(model, args, out_dir / "world.pt")
    if args.stage in {"detector", "both"}:
        if args.stage == "detector" and not args.world_path:
            raise SystemExit("detector stage needs --world_path from the reference-fitting run")
        train_detector(model, det_records, eval_records, args, device)
        checkpoint(model, args, out_dir / "detector.pt")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
