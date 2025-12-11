# 本文件实现 Spectral Consensus 聚合与 LoRA 谱分析
# 服务器将利用文本模态评估权重，指导视觉模态鲁棒聚合
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Spectral Consensus Aggregation")
    parser.add_argument("--config", type=str, default="configs/local.yaml")
    parser.add_argument("--snapshots", type=str, nargs="+", help="需要聚合的客户端 LoRA 快照文件")
    parser.add_argument("--strategy", type=str, default="spectral", choices=["spectral", "fedavg"])
    parser.add_argument("--top_k", type=int, default=4, help="谱空间比较的主成分数量")
    parser.add_argument("--temperature", type=float, default=5.0, help="Softmax 温度系数")
    parser.add_argument("--output", type=str, default="checkpoints/global_spectral.pt", help="保存聚合结果")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_delta_w(lora_A: torch.Tensor, lora_B: torch.Tensor) -> torch.Tensor:
    """计算 LoRA 更新量 ΔW = B @ A。"""
    return lora_B @ lora_A


def compute_subspace_similarity(delta_local: torch.Tensor, delta_global: torch.Tensor, top_k: int = 4) -> float:
    """利用前 k 个主成分的重叠度衡量谱一致性。"""
    delta_local = delta_local.float()
    delta_global = delta_global.float()
    try:
        u_loc, _, _ = torch.linalg.svd(delta_local, full_matrices=False)
        u_glo, _, _ = torch.linalg.svd(delta_global, full_matrices=False)
    except RuntimeError:
        return 0.0

    k = min(top_k, u_loc.shape[1], u_glo.shape[1])
    if k == 0:
        return 0.0
    u_loc_top = u_loc[:, :k]
    u_glo_top = u_glo[:, :k]
    similarity = torch.norm(u_glo_top.T @ u_loc_top) ** 2 / k
    return similarity.item()


def execute_spectral_aggregation(
    client_states: List[Dict[str, torch.Tensor]],
    top_k: int = 4,
    temperature: float = 5.0,
) -> Dict[str, torch.Tensor]:
    """完整的 Spectral Consensus 聚合流程。"""
    if not client_states:
        raise ValueError("client_states 不能为空")

    num_clients = len(client_states)
    keys = list(client_states[0].keys())

    text_modules = {
        k.replace(".lora_A.weight", "")
        for k in keys
        if "text_model" in k and k.endswith(".lora_A.weight")
    }

    if not text_modules:
        return aggregate_mean(client_states)

    global_text_deltas: Dict[str, torch.Tensor] = {}
    for mod in text_modules:
        key_a = f"{mod}.lora_A.weight"
        key_b = f"{mod}.lora_B.weight"
        avg_delta = sum(get_delta_w(c[key_a], c[key_b]) for c in client_states) / num_clients
        global_text_deltas[mod] = avg_delta

    scores = []
    for client in client_states:
        sim_sum = 0.0
        for mod in text_modules:
            key_a = f"{mod}.lora_A.weight"
            key_b = f"{mod}.lora_B.weight"
            delta_local = get_delta_w(client[key_a], client[key_b])
            sim_sum += compute_subspace_similarity(delta_local, global_text_deltas[mod], top_k=top_k)
        scores.append(sim_sum / max(1, len(text_modules)))

    score_tensor = torch.tensor(scores)
    weights = F.softmax(score_tensor * temperature, dim=0)

    aggregated: Dict[str, torch.Tensor] = {}
    for key in keys:
        params = torch.stack([client[key].float() for client in client_states], dim=0)
        base_dtype = client_states[0][key].dtype
        if "vision_model" in key:
            view_shape = (weights.shape[0],) + (1,) * (params.ndim - 1)
            weighted = (params * weights.view(view_shape)).sum(dim=0)
            aggregated[key] = weighted.to(base_dtype)
        else:
            aggregated[key] = params.mean(dim=0).to(base_dtype)
    return aggregated


def aggregate_mean(client_states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """简单平均作为回退策略。"""
    if not client_states:
        raise ValueError("client_states 不能为空")
    keys = client_states[0].keys()
    return {k: torch.stack([c[k] for c in client_states], dim=0).mean(dim=0) for k in keys}


class ServerCoordinator:
    """负责加载客户端快照并执行指定的聚合策略。"""

    def __init__(self, cfg: Dict) -> None:
        server_cfg = cfg.get("server", {})
        gpu_id = server_cfg.get("gpu", 0)
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(gpu_id)
        else:
            self.device = torch.device("cpu")
        self.cfg = cfg

    def load_lora_state(self, path: str) -> Dict[str, torch.Tensor]:
        payload = torch.load(path, map_location=self.device)
        return payload["state_dict"]

    def spectral_consensus(self, states: List[Dict[str, torch.Tensor]], top_k: int, temperature: float):
        return execute_spectral_aggregation(states, top_k=top_k, temperature=temperature)

    def fedavg(self, states: List[Dict[str, torch.Tensor]]):
        return aggregate_mean(states)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    coordinator = ServerCoordinator(cfg)

    if not args.snapshots:
        raise ValueError("请通过 --snapshots 指定至少一个 LoRA 快照文件")

    states = [coordinator.load_lora_state(p) for p in args.snapshots]

    if args.strategy == "spectral":
        aggregated = coordinator.spectral_consensus(states, top_k=args.top_k, temperature=args.temperature)
    else:
        aggregated = coordinator.fedavg(states)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": aggregated}, output_path)
    print(f"已保存聚合结果至 {output_path}")


if __name__ == "__main__":
    main()