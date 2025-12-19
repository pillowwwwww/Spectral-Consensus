import argparse
import copy
import math
from pathlib import Path

import torch
import yaml

# 让脚本在项目根目录/任意目录运行都能 import src.*
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.strategies import fedavg_strategy, spectral_merging_b_strategy


def _extract_state_dict(obj):
    if isinstance(obj, dict) and isinstance(obj.get("state_dict"), dict):
        return obj["state_dict"]
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="如 checkpoints/20251213_175445_checkpoints")
    parser.add_argument("--config", type=str, default="src/configs/fed_ours_b.yaml")
    parser.add_argument("--device", type=str, default=None, help="如 cuda:0 或 cpu；默认用 config 里的 train.device")
    parser.add_argument("--prune_ratio", type=float, default=0.0, help="用于 ours_b 的 prune_ratio（默认 0）")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg2 = copy.deepcopy(cfg)
    cfg2.setdefault("train", {})["prune_ratio"] = float(args.prune_ratio)

    device_str = args.device or cfg.get("train", {}).get("device", "cpu")
    device = torch.device(device_str)

    run_dir = Path(args.run_dir)
    client_domains = [c["domain"] for c in cfg["clients"]]
    client_ckpts = [
        run_dir / "clients" / d.replace(" ", "_") / "round_001.pt" for d in client_domains
    ]

    client_states = []
    for p in client_ckpts:
        ckpt = torch.load(p, map_location="cpu")
        client_states.append(_extract_state_dict(ckpt))

    fedavg_global = fedavg_strategy(client_states, device=device, cfg=cfg)
    oursb0_global = spectral_merging_b_strategy(client_states, device=device, cfg=cfg2)

    keys = sorted(set(fedavg_global) & set(oursb0_global))
    print("common keys:", len(keys))
    print("missing in ours_b:", len(set(fedavg_global) - set(oursb0_global)))
    print("missing in fedavg:", len(set(oursb0_global) - set(fedavg_global)))

    sq = 0.0
    mx = 0.0
    for k in keys:
        a = fedavg_global[k]
        b = oursb0_global[k]
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            continue
        d = a.float().cpu() - b.float().cpu()
        sq += float((d * d).sum().item())
        mx = max(mx, float(d.abs().max().item()))

    print("L2 diff:", math.sqrt(sq))
    print("Max abs diff:", mx)


if __name__ == "__main__":
    main()
