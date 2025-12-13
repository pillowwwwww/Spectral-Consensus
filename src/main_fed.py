# src/main_fed.py
import argparse
import sys
from pathlib import Path

import swanlab

# 将项目根目录加入路径，防止 src 内部 import 报错
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.server import FedServer


def main():
    parser = argparse.ArgumentParser(description="Run Federated Learning Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/fed.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help=(
            "Override train.strategy in the config "
            "(e.g. fedavg, spectral_merging, spectral_merging_b)."
        ),
    )
    args = parser.parse_args()

    # 初始化并运行联邦服务器，所有的逻辑控制都在 Server 类里，这里只是一个启动按键
    server = FedServer(args.config, strategy_override=args.strategy)

    # === SwanLab: 初始化一个 run，用于跟踪本次联邦实验（包括 fedavg 与 ours_b） ===
    train_cfg = server.cfg.get("train", {})
    lora_cfg = server.cfg.get("lora", {})

    swanlab.init(
        project="myfedclip",
        workspace="chang",
        config={
            "config_path": args.config,
            "strategy": train_cfg.get("strategy", "fedavg"),
            "rounds": train_cfg.get("rounds"),
            "epochs_per_round": train_cfg.get("epochs_per_round"),
            "model_name": train_cfg.get("model_name"),
            "prune_ratio": train_cfg.get("prune_ratio"),
            "batch_size": train_cfg.get("batch_size"),
            "learning_rate": train_cfg.get("learning_rate"),
            "lora_r": lora_cfg.get("r"),
        },
    )
    # === SwanLab init 结束 ===

    try:
        server.run()
    finally:
        # 始终结束 SwanLab 运行（即使中途出错）
        swanlab.finish()


if __name__ == "__main__":
    main()

