# 本文件负责驱动各个客户端的 LoRA 本地训练
# 通过读取配置启动 4 个域的独立训练并记录 LoRA 轨迹
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
import random
import numpy as np
import torch
from data_loader import create_officehome_dataloader
from learner import ClipLoRALearner, LearnerConfig
from utils.logger import create_tb_writer, dump_jsonl, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train Office-Home clients with CLIP+LoRA")
    parser.add_argument("--config", type=str, default="configs/local.yaml", help="配置文件路径")
    parser.add_argument("--domains", type=str, nargs="*", help="指定需要训练的域，默认全部")
    parser.add_argument("--max_steps", type=int, default=None, help="调试用，限制每个客户端训练步数")
    return parser.parse_args()

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_domain_key(domain: str) -> str:
    return domain.replace(" ", "_").lower()


def train_single_client(domain: str, cfg: Dict, max_steps_override: Optional[int]) -> None:
   
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    lora_cfg = cfg["lora"]
    client_mapping = cfg["client_mapping"]

    key = normalize_domain_key(domain)
    if key not in client_mapping:
        raise KeyError(f"配置文件中缺少 {key} 的 GPU 映射")
    gpu_id = int(client_mapping[key])

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    run_name = f"{domain.replace(' ', '_').lower()}"
    save_root = Path(train_cfg["save_dir"]) / run_name
    save_root.mkdir(parents=True, exist_ok=True)
    log_dir = Path(train_cfg["log_dir"]) / run_name
    logger = setup_logger(f"client_{run_name}", log_dir.as_posix())
    writer = create_tb_writer(train_cfg["log_dir"], run_name)

    dataloader, dataset_info = create_officehome_dataloader(
        root=data_cfg["office_home_root"],
        domain=domain,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        shuffle=True,
    )
    class_names = sorted(dataset_info.class_to_idx, key=dataset_info.class_to_idx.get)
    learner_cfg = LearnerConfig(
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        grad_clip=train_cfg.get("grad_clip", 1.0),
        logging_steps=train_cfg["logging_steps"],
        snapshot_steps=train_cfg.get("snapshot_steps", train_cfg["logging_steps"]),
        fp16=train_cfg.get("fp16", True),
    )
    seed_everything(42)
    learner = ClipLoRALearner(
        class_names=class_names,
        device=device,
        lora_cfg=lora_cfg,
        learner_cfg=learner_cfg,
        output_dir=save_root.as_posix(),
    )

    total_epochs = train_cfg["epochs"]
    max_steps = max_steps_override or train_cfg.get("max_train_steps") or float("inf")

    logger.info("启动客户端 %s：samples=%d, classes=%d, device=%s", domain, dataset_info.num_samples, dataset_info.num_classes, device)

    snapshot_every = train_cfg.get("snapshot_steps", 100)
    global_step = 0

    for epoch in range(1, total_epochs + 1):
        for batch in dataloader:
            metrics = learner.train_step(batch)
            global_step = learner.global_step

            if global_step % learner_cfg.logging_steps == 0:
                log_payload = {"epoch": epoch, "step": global_step, **metrics}
                dump_jsonl(log_payload, (log_dir / "metrics.jsonl").as_posix())
                writer.add_scalar("train/loss", metrics["loss"], global_step)
                writer.add_scalar("train/acc", metrics["acc"], global_step)
                logger.info("Epoch %d Step %d | loss %.4f | acc %.4f", epoch, global_step, metrics["loss"], metrics["acc"])

            if global_step % snapshot_every == 0:
                learner.save_lora_snapshot(f"{run_name}_step{global_step:06d}")

            if global_step % train_cfg.get("checkpoint_steps", snapshot_every * 5) == 0:
                learner.save_checkpoint(f"{run_name}_step{global_step:06d}.pt", extra={"domain": domain})

            if global_step >= max_steps:
                logger.info("达到 max_steps=%s，提前停止。", max_steps)
                break

        if global_step >= max_steps:
            break

    # 保存最终检查点
    learner.save_checkpoint(f"{run_name}_final.pt", extra={"domain": domain})
    writer.close()
    logger.info("客户端 %s 训练完成，总步数 %d", domain, global_step)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    target_domains: List[str]
    if args.domains:
        target_domains = args.domains
    else:
        target_domains = cfg.get("domains", ["Art", "Clipart", "Product", "Real World"])

    os.makedirs(cfg["train"]["save_dir"], exist_ok=True)
    os.makedirs(cfg["train"]["log_dir"], exist_ok=True)

    for domain in target_domains:
        train_single_client(domain, cfg, args.max_steps)


if __name__ == "__main__":
    main()

