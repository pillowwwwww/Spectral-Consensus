#构建 CLIP+LoRA + anchor dataloader（需要读 YAML）
from __future__ import annotations

import copy
from typing import Callable, Dict, List

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, CLIPModel

from src.random_pruning import RandomPruningAggregator
from src.spectral_merging import SpectralAggregator as SpectralAggregatorA
from src.spectral_merging_b import SensitivityAggregator
from src.anchor_loader import (
    build_coco_anchor_dataloader,
    image_root as DEFAULT_ANCHOR_IMAGE_ROOT,
    captions_json as DEFAULT_ANCHOR_CAPTIONS_JSON,
)


StateDict = Dict[str, torch.Tensor]
StrategyFn = Callable[[List[StateDict], torch.device, Dict], StateDict]


def _fedavg_impl(
    client_state_dicts: List[StateDict],
    device: torch.device,
) -> StateDict:
    """Standard FedAvg over a list of client LoRA state_dicts."""
    if not client_state_dicts:
        raise ValueError("client_state_dicts must not be empty for FedAvg.")

    num_clients = len(client_state_dicts)
    avg_state_dict: StateDict = copy.deepcopy(client_state_dicts[0])

    for key, value in avg_state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue

        summed_param = client_state_dicts[0][key].to(device)
        for client_idx in range(1, num_clients):
            summed_param = summed_param + client_state_dicts[client_idx][key].to(device)

        avg_state_dict[key] = summed_param / float(num_clients)

    return avg_state_dict


def fedavg_strategy(
    client_state_dicts: List[StateDict],
    device: torch.device,
    cfg: Dict,
) -> StateDict:
    """Public entry for FedAvg strategy."""
    del cfg  # cfg is unused for pure FedAvg but kept for unified signature
    return _fedavg_impl(client_state_dicts, device=device)


def spectral_merging_strategy(
    client_state_dicts: List[StateDict],
    device: torch.device,
    cfg: Dict,
) -> StateDict:
    """Strategy that uses spectral_merging.SpectralAggregator (A 版谱聚合)。"""
    temperature = cfg.get("train", {}).get("temperature", 0.1)
    aggregator = SpectralAggregatorA(device=device, temperature=temperature)
    return aggregator._spectral_weighted_merge(client_state_dicts)  # type: ignore[attr-defined]

## 懒加载
_OURS_B_AGGREGATOR: SensitivityAggregator | None = None


def _build_ours_b_aggregator(device: torch.device, cfg: Dict) -> SensitivityAggregator:
    """
    在 server 端构建 CLIP+LoRA 的 PeftModel，并基于 COCO 构建 anchor_dataloader，返回 B 版聚合器。
    """
    train_cfg = cfg.get("train", {})
    lora_cfg = cfg.get("lora", {})
    anchor_cfg = cfg.get("anchor", {})
    clients_cfg = cfg.get("clients", [])

    model_name = train_cfg["model_name"]

    # 1. 构建基座 CLIP 模型与 tokenizer
    base_model = CLIPModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. 构建与客户端完全一致的 LoRA 配置，并包装成 PeftModel
    lora_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias="none",
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj"]),
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.to(device)
    peft_model.eval()

    # 3. 构建 COCO anchor_dataloader（可从 YAML 的 anchor 段覆盖路径与超参）
    anchor_image_root = anchor_cfg.get("image_root", DEFAULT_ANCHOR_IMAGE_ROOT)
    anchor_captions_json = anchor_cfg.get("captions_json", DEFAULT_ANCHOR_CAPTIONS_JSON)
    anchor_batch_size = anchor_cfg.get("batch_size", 32)

    num_images_cfg = anchor_cfg.get("num_images", (32, 64))
    if isinstance(num_images_cfg, list):
        num_images = tuple(num_images_cfg)
    else:
        num_images = num_images_cfg

    anchor_loader = build_coco_anchor_dataloader(
        tokenizer=tokenizer,
        image_root=anchor_image_root,
        captions_json=anchor_captions_json,
        batch_size=anchor_batch_size,
        num_images=num_images,
        num_workers=anchor_cfg.get("num_workers", 4),
        seed=anchor_cfg.get("seed", 42),
        shuffle=anchor_cfg.get("shuffle", False),
    )

    prune_ratio = train_cfg.get("prune_ratio", 0.7)

    # 客户端域名列表，用于在 B 版中做 Server/Anchor_Loss_{Domain} 等 SwanLab 指标
    client_domains = [
        client.get("domain", f"Client_{idx}") for idx, client in enumerate(clients_cfg)
    ]

    return SensitivityAggregator(
        model=peft_model,
        anchor_dataloader=anchor_loader,
        device=device,
        prune_ratio=prune_ratio,
        client_domains=client_domains,
    )


def spectral_merging_b_strategy(
    client_state_dicts: List[StateDict],
    device: torch.device,
    cfg: Dict,
) -> StateDict:
    """
    B 版谱聚合策略：
    - 在 server 端构建一次 CLIP+LoRA PeftModel + COCO 锚点 DataLoader；
    - 对每轮接收到的客户端 LoRA ΔW 做敏感度剪枝 + AvgMerge。
    """
    global _OURS_B_AGGREGATOR

    if _OURS_B_AGGREGATOR is None:
        _OURS_B_AGGREGATOR = _build_ours_b_aggregator(device, cfg)

    return _OURS_B_AGGREGATOR.aggregate(client_state_dicts)


def random_pruning_strategy(
    client_state_dicts: List[StateDict],
    device: torch.device,
    cfg: Dict,
) -> StateDict:
    """Randomly prune LoRA parameters before FedAvg aggregation."""
    train_cfg = cfg.get("train", {})
    prune_ratio = train_cfg.get("prune_ratio", 0.5)
    seed = train_cfg.get("random_prune_seed", train_cfg.get("seed"))

    aggregator = RandomPruningAggregator(
        prune_ratio=prune_ratio,
        seed=seed,
        device=device,
    )
    return aggregator.aggregate(client_state_dicts)


STRATEGY_REGISTRY: Dict[str, StrategyFn] = {
    # 经典基线
    "fedavg": fedavg_strategy,

    # 原始谱聚合实现（A 版）
    "ours": spectral_merging_strategy,
    "spectral_merging": spectral_merging_strategy,

    # 变体 B（对应 spectral_merging_b.py 中的敏感度剪枝聚合）
    "ours_b": spectral_merging_b_strategy,
    "spectral_merging_b": spectral_merging_b_strategy,

    # 随机剪枝基线
    "random_pruning": random_pruning_strategy,
}
