# 本文件提供统一的日志与指标记录工具
# 用于客户端训练与服务器分析阶段的复用
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from torch.utils.tensorboard import SummaryWriter


def setup_logger(name: str, log_dir: str, filename: str = "runtime.log") -> logging.Logger:
    """创建基础 logger 并输出到文件与控制台。"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(Path(log_dir) / filename, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def create_tb_writer(log_dir: str, run_name: str) -> SummaryWriter:
    """初始化 TensorBoard SummaryWriter。"""
    run_path = Path(log_dir) / run_name
    run_path.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(run_path.as_posix())


def log_metrics(logger: logging.Logger, writer: Optional[SummaryWriter], step: int, metrics: Dict[str, float]) -> None:
    """统一打印并写入 TensorBoard。"""
    msg = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    logger.info("Step %s | %s", step, msg)
    if writer is not None:
        for key, value in metrics.items():
            writer.add_scalar(key, value, step)


def dump_jsonl(data: Dict, output_path: str) -> None:
    """以 jsonl 形式追加记录，方便后续分析。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

