# 本文件实现联邦消融实验主控脚本
# 可模拟含恶意客户端的多轮谱共识聚合流程
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

import torch
import yaml
from torch.utils.data import DataLoader

from .data_loader import create_officehome_dataloader
from .learner import ClipLoRALearner, LearnerConfig
from .server import aggregate_mean, execute_spectral_aggregation


def parse_args():
    parser = argparse.ArgumentParser(description="Federated Orchestration with Spectral Consensus")
    parser.add_argument("--config", type=str, default="configs/local.yaml", help="配置文件路径")
    parser.add_argument("--rounds", type=int, default=5, help="联邦轮数")
    parser.add_argument("--local_steps", type=int, default=200, help="每轮每个客户端的训练步数")
    parser.add_argument("--strategy", type=str, default="spectral", choices=["spectral", "fedavg"])
    parser.add_argument("--malicious_domains", type=str, nargs="*", default=["real_world"], help="启用语义攻击的域")
    parser.add_argument("--top_k", type=int, default=4, help="谱空间比较的主成分数量")
    parser.add_argument("--temperature", type=float, default=5.0, help="Softmax 温度系数")
    parser.add_argument("--save_dir", type=str, default="checkpoints/global_rounds", help="保存聚合结果路径")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_domain_key(domain: str) -> str:
    return domain.replace(" ", "_").lower()


@dataclass
class ClientContext:
    domain: str
    class_names: Sequence[str]
    dataloader: DataLoader
    gpu_id: int
    malicious: bool
    iterator: Iterator = field(default=None, init=False)

    def next_batch(self):
        if self.iterator is None:
            self.iterator = iter(self.dataloader)
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)


def build_clients(cfg: Dict, malicious_domains: Sequence[str]) -> List[ClientContext]:
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    mapping = cfg["client_mapping"]

    malicious_set = {normalize_domain_key(d) for d in malicious_domains}
    target_domains = cfg.get("domains", ["Art", "Clipart", "Product", "Real World"])

    clients: List[ClientContext] = []
    for domain in target_domains:
        loader, info = create_officehome_dataloader(
            root=data_cfg["office_home_root"],
            domain=domain,
            batch_size=train_cfg["batch_size"],
            num_workers=train_cfg["num_workers"],
            shuffle=True,
        )
        class_names = sorted(info.class_to_idx, key=info.class_to_idx.get)
        key = normalize_domain_key(domain)
        gpu_id = int(mapping[key])
        clients.append(
            ClientContext(
                domain=domain,
                class_names=class_names,
                dataloader=loader,
                gpu_id=gpu_id,
                malicious=key in malicious_set,
            )
        )
    return clients


def build_learner_config(train_cfg: Dict) -> LearnerConfig:
    return LearnerConfig(
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        grad_clip=train_cfg.get("grad_clip", 1.0),
        logging_steps=train_cfg.get("logging_steps", 100),
        snapshot_steps=train_cfg.get("snapshot_steps", 100),
        fp16=train_cfg.get("fp16", True),
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)

    clients = build_clients(cfg, args.malicious_domains)
    learner_cfg = build_learner_config(cfg["train"])
    lora_cfg = cfg["lora"]

    device_available = torch.cuda.is_available()
    global_state: Dict[str, torch.Tensor] = {}

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for round_id in range(1, args.rounds + 1):
        client_states: List[Dict[str, torch.Tensor]] = []
        print(f"\n======= Round {round_id} / {args.rounds} =======")

        for ctx in clients:
            device = torch.device(f"cuda:{ctx.gpu_id}") if device_available else torch.device("cpu")
            learner = ClipLoRALearner(
                class_names=ctx.class_names,
                device=device,
                lora_cfg=lora_cfg,
                learner_cfg=learner_cfg,
                output_dir=f"checkpoints/{normalize_domain_key(ctx.domain)}",
                is_malicious=ctx.malicious,
            )

            if global_state:
                learner.load_lora_state(global_state)

            last_metrics = {}
            for _ in range(args.local_steps):
                batch = ctx.next_batch()
                last_metrics = learner.train_step(batch)

            client_state = learner.extract_lora_delta("all")
            client_states.append(client_state)
            status = "MAL" if ctx.malicious else "BEN"
            print(
                f"[{ctx.domain:<10}] {status} | loss={last_metrics.get('loss', 0):.4f} "
                f"acc={last_metrics.get('acc', 0):.4f}"
            )

        if args.strategy == "spectral":
            global_state = execute_spectral_aggregation(
                client_states,
                top_k=args.top_k,
                temperature=args.temperature,
            )
        else:
            global_state = aggregate_mean(client_states)

        round_path = save_dir / f"round_{round_id:02d}.pt"
        torch.save({"round": round_id, "state_dict": global_state}, round_path)
        print(f"Round {round_id} 聚合完成，结果已保存到 {round_path}")


if __name__ == "__main__":
    main()

