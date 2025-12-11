import os
import copy
import logging
from datetime import datetime
from pathlib import Path
from typing import List

import swanlab
import torch
import yaml

from src.learner import ClipLoRALearner, LearnerConfig
from src.data_loader import create_officehome_dataloader
from src.strategies import STRATEGY_REGISTRY


# 配置基础日志，具体文件路径由 FedServer 在构造函数中追加 FileHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("FedServer")


class FedServer:
    def __init__(self, config_path: str, strategy_override: str | None = None):
        self.cfg = self._load_config(config_path)
        if strategy_override is not None:
            self.cfg.setdefault("train", {})["strategy"] = strategy_override

        # 项目根目录 (myFedCLIP) 和本次实验的 run_id
        self.root_dir = Path(__file__).resolve().parents[1]
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # checkpoints/<run_id>_checkpoints/{global,clients}
        self.ckpt_root = self.root_dir / "checkpoints" / f"{self.run_id}_checkpoints"
        self.global_ckpt_dir = self.ckpt_root / "global"
        self.client_ckpt_dir = self.ckpt_root / "clients"
        self.global_ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.client_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # logs/fed_server_<run_id>.log
        self.log_dir = self.root_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.log_dir / f"fed_server_{self.run_id}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info("FedServer run_id=%s", self.run_id)
        logger.info("Checkpoints root: %s", self.ckpt_root.as_posix())
        logger.info("Log file: %s", log_file.as_posix())

        self.device = torch.device(self.cfg["train"].get("device", "cuda"))

        strategy_name = self.cfg["train"].get("strategy", "fedavg")
        if strategy_name not in STRATEGY_REGISTRY:
            raise ValueError(
                f"Unknown strategy '{strategy_name}'. "
                f"Available: {list(STRATEGY_REGISTRY.keys())}"
            )
        self.strategy_name = strategy_name
        self.strategy_fn = STRATEGY_REGISTRY[strategy_name]

        # 准备客户端
        self.clients: List[ClipLoRALearner] = []
        self.client_loaders = []
        self.client_domains: List[str] = []
        self._setup_clients()

        # 初始化全局模型 (取第一个客户端的 LoRA 参数作为初始全局状态)
        self.global_model_state = copy.deepcopy(
            self.clients[0].extract_lora_delta("all")
        )

        logger.info(
            "Server initialized with %d clients. Strategy: %s",
            len(self.clients),
            self.strategy_name,
        )

    def _load_config(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _setup_clients(self) -> None:
        """初始化所有客户端实例和数据加载器。"""
        data_cfg = self.cfg["data"]
        train_cfg = self.cfg["train"]
        lora_cfg = self.cfg["lora"]
        client_list = self.cfg["clients"]  # list of {domain: str, malicious: bool}

        for client_info in client_list:
            domain = client_info["domain"]
            is_malicious = client_info.get("malicious", False)

            # 1. 准备数据：训练阶段四个域的 shuffle=True
            dataloader, dataset_info = create_officehome_dataloader(
                root=data_cfg["office_home_root"],
                domain=domain,
                batch_size=train_cfg["batch_size"],
                num_workers=4,
                shuffle=True,
            )
            self.client_loaders.append(dataloader)
            self.client_domains.append(domain)

            # 为该客户端准备专属 checkpoint 目录
            client_ckpt_root = self.client_ckpt_dir / domain.replace(" ", "_")
            client_ckpt_root.mkdir(parents=True, exist_ok=True)

            # 2. 准备 Learner
            class_names = sorted(
                dataset_info.class_to_idx, key=dataset_info.class_to_idx.get
            )

            learner_cfg = LearnerConfig(
                model_name=train_cfg["model_name"],
                learning_rate=train_cfg["learning_rate"],
                weight_decay=train_cfg["weight_decay"],
                fp16=train_cfg["fp16"],
            )

            learner = ClipLoRALearner(
                class_names=class_names,
                device=self.device,
                lora_cfg=lora_cfg,
                learner_cfg=learner_cfg,
                output_dir=client_ckpt_root.as_posix(),
                is_malicious=is_malicious,
                poison_shuffle_prob=1.0 if is_malicious else 0.0,
            )
            self.clients.append(learner)
            logger.info("Client [%s] initialized. Malicious: %s", domain, is_malicious)

    def run(self) -> None:
        rounds = self.cfg["train"]["rounds"]
        epochs_per_round = self.cfg["train"]["epochs_per_round"]

        for round_idx in range(rounds):
            logger.info("======== Round %d / %d ========", round_idx + 1, rounds)

            client_weights = []
            client_losses = []

            # --- 1. 客户端本地训练 ---
            for client_idx, client in enumerate(self.clients):
                # A. 下发全局参数 (加载 LoRA 权重)
                client.load_lora_state(self.global_model_state)

                # B. 本地训练
                loader = self.client_loaders[client_idx]
                loss_avg = 0.0
                steps = 0

                # 简单起见，每个 Round 训练固定的步数 / Epoch
                for _ in range(epochs_per_round):
                    for batch in loader:
                        metrics = client.train_step(batch)
                        loss_avg += metrics["loss"]
                        steps += 1

                loss_avg /= max(steps, 1)
                client_losses.append(loss_avg)

                # C. 提取训练后的参数 (只提取 LoRA 部分)
                client_weights.append(client.extract_lora_delta("all"))

                logger.info("  Client %d Train Loss: %.4f", client_idx, loss_avg)

                # === SwanLab: 按客户端记录本轮训练 loss (fedavg + ours_b 均有) ===
                try:
                    domain = self.client_domains[client_idx]
                    swanlab.log(
                        {
                            "round": round_idx + 1,
                            f"Client/Train_Loss/{domain}": float(loss_avg),
                        }
                    )
                except Exception:
                    # 若 SwanLab 未初始化或出现其他错误，这里静默跳过
                    pass
                # === SwanLab log 结束 ===

                # D. 保存该 round 的客户端 checkpoint
                domain = self.client_domains[client_idx]
                ckpt_name = f"round_{round_idx + 1:03d}.pt"
                client.save_checkpoint(
                    ckpt_name,
                    extra={"domain": domain, "round": round_idx + 1},
                )

            # --- 2. 服务器聚合 ---
            logger.info(
                "Aggregating parameters with strategy=%s ...", self.strategy_name
            )

            # === SwanLab: 记录每轮的平均 client loss (fedavg + ours_b 均有) ===
            if client_losses:
                try:
                    avg_loss = sum(client_losses) / len(client_losses)
                    swanlab.log(
                        {
                            "round": round_idx + 1,
                            "Client/Train_Loss/avg_clients": float(avg_loss),
                        }
                    )
                except Exception:
                    pass
            # === SwanLab avg loss log 结束 ===

            # 统一通过 STRATEGY_REGISTRY 中注册的实现进行聚合
            self.global_model_state = self.strategy_fn(
                client_weights,
                self.device,
                self.cfg,
            )

            # --- 3. (可选) 评估 / 保存全局模型 ---
            if (round_idx + 1) % 5 == 0:
                self._save_global_model(round_idx + 1)

    def _save_global_model(self, round_num: int) -> None:
        path = self.global_ckpt_dir / f"global_round_{round_num}.pt"
        torch.save(self.global_model_state, path)
        logger.info("Saved global model to %s", path)

