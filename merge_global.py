"""
Merge-only script to generate multiple global checkpoints from a fixed run_id.
Usage (example):
    python merge_global.py --run_id 20251214_162336 \
        --config src/configs/fed_ours_b.yaml \
        --prune_ratios 0 0.01 0.05 0.1 0.2 0.3

This will read clients/<domain>/round_001.pt under the given run_id once,
then run the configured strategy (spectral_merging_b by default) with
different prune_ratio values, and save merged globals under:
    checkpoints/<run_id>_checkpoints/global/merge_round_001_pruneX.pt

For random_pruning sweeps, add seeds if needed:
    python merge_global.py --run_id 20251214_162336 --strategy random_pruning \\
        --prune_ratios 0.9 0.95 0.99 --seeds 1 2 3
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Dict, List

import torch
import yaml

# ensure src is importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from src.strategies import (
    fedavg_strategy,
    random_pruning_strategy,
    spectral_merging_strategy,
    spectral_merging_b_strategy,
)
import src.strategies as strategies_module  # for resetting _OURS_B_AGGREGATOR cache


StateDict = Dict[str, torch.Tensor]


def load_client_states(run_dir: Path, domains: List[str], round_id: int) -> List[StateDict]:
    client_states: List[StateDict] = []
    for domain in domains:
        ckpt_path = run_dir / "clients" / domain.replace(" ", "_") / f"round_{round_id:03d}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Client checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and isinstance(ckpt.get("state_dict"), dict):
            client_states.append(ckpt["state_dict"])
        else:
            client_states.append(ckpt)
    return client_states


def format_prune_ratio(r: float) -> str:
    # safe filename fragment, e.g., 0.1 -> 0p1, 0 -> 0, 0.05 -> 0p05
    s = f"{r}"
    return s.replace(".", "p")


def merge_once(
    client_states: List[StateDict],
    cfg: Dict,
    device: torch.device,
    strategy: str,
) -> StateDict:
    strategy = strategy.lower()
    if strategy == "fedavg":
        return fedavg_strategy(client_states, device=device, cfg=cfg)
    elif strategy in {"spectral_merging", "ours"}:
        return spectral_merging_strategy(client_states, device=device, cfg=cfg)
    elif strategy in {"spectral_merging_b", "ours_b"}:
        # reset cached aggregator so prune_ratio changes take effect
        if hasattr(strategies_module, "_OURS_B_AGGREGATOR"):
            strategies_module._OURS_B_AGGREGATOR = None  # type: ignore[attr-defined]
        return spectral_merging_b_strategy(client_states, device=device, cfg=cfg)
    elif strategy == "random_pruning":
        return random_pruning_strategy(client_states, device=device, cfg=cfg)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge fixed client checkpoints with different prune_ratio values (supports spectral_merging_b / random_pruning sweeps)."
    )
    parser.add_argument("--run_id", type=str, required=True, help="e.g., 20251214_162336")
    parser.add_argument("--config", type=str, default="src/configs/fed_ours_b.yaml", help="Config file path")
    parser.add_argument("--round", type=int, default=1, help="Which round_* client ckpt to merge")
    parser.add_argument(
        "--prune_ratios",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.05, 0.1, 0.2, 0.3],
        help="List of prune_ratio values to try (used when strategy is spectral_merging_b or random_pruning)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Override train.strategy; default uses config.train.strategy",
    )
    parser.add_argument("--device", type=str, default=None, help="Merge device, e.g., cuda:0 or cpu")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional seeds for random_pruning; if provided, merges run for each seed.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    train_cfg = cfg.setdefault("train", {})

    strategy = args.strategy or train_cfg.get("strategy", "spectral_merging_b")
    device_str = args.device or train_cfg.get("device", "cpu")
    device = torch.device(device_str)

    run_dir = ROOT / "checkpoints" / f"{args.run_id}_checkpoints"
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    domains = [c["domain"] for c in cfg.get("clients", [])]
    if not domains:
        raise ValueError("No clients specified in config.")

    print(f"Using run_id={args.run_id} (round {args.round:03d}), strategy={strategy}, device={device}")
    client_states = load_client_states(run_dir, domains, args.round)
    out_dir = run_dir / "global"
    out_dir.mkdir(parents=True, exist_ok=True)

    strat_lower = strategy.lower()
    ratios = args.prune_ratios if strat_lower in {"spectral_merging_b", "ours_b", "random_pruning"} else [None]
    seeds = args.seeds if strat_lower == "random_pruning" and args.seeds else [None]

    for r in ratios:
        for seed in seeds:
            cfg_tmp = copy.deepcopy(cfg)
            train_tmp = cfg_tmp.setdefault("train", {})
            name_parts = [f"merge_round_{args.round:03d}"]

            if r is not None:
                train_tmp["prune_ratio"] = float(r)
                tag = format_prune_ratio(r)
                name_parts.append(f"prune{tag}")

            if seed is not None and strat_lower == "random_pruning":
                train_tmp["random_prune_seed"] = int(seed)
                name_parts.append(f"seed{seed}")

            if strat_lower == "fedavg":
                out_path = out_dir / f"{name_parts[0]}_fedavg.pt"
            else:
                out_path = out_dir / ("_".join(name_parts) + ".pt")

            print(f"\n[Merge] strategy={strategy}, prune_ratio={r}, seed={seed} -> {out_path.name}")
            merged = merge_once(client_states, cfg_tmp, device=device, strategy=strategy)
            torch.save(merged, out_path)
            print(f"Saved merged global to: {out_path}")


if __name__ == "__main__":
    main()
