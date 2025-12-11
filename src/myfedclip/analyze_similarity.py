# 本文件实现 LoRA 文本/视觉分支的余弦相似度分析
# 用于生成 Figure 1，验证视觉发散与文本稳定假设
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

SNAPSHOT_PATTERN = re.compile(r"step(\d+)")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze LoRA similarity between clients")
    parser.add_argument("--client_a", type=str, required=True, help="客户端 A 的快照目录")
    parser.add_argument("--client_b", type=str, required=True, help="客户端 B 的快照目录")
    parser.add_argument("--output_csv", type=str, default="outputs/similarity.csv")
    parser.add_argument("--output_fig", type=str, default="outputs/similarity.png")
    return parser.parse_args()


def list_snapshots(directory: str) -> Dict[int, Path]:
    """扫描目录下的 lora_*.pt，并返回 step -> path 的映射。"""
    path = Path(directory)
    files = sorted(path.glob("lora_*.pt"))
    mapping: Dict[int, Path] = {}
    for file in files:
        if "step" in file.name:
            match = SNAPSHOT_PATTERN.search(file.name)
            if match:
                step = int(match.group(1))
            else:
                payload = torch.load(file, map_location="cpu")
                step = int(payload.get("global_step", -1))
        else:
            payload = torch.load(file, map_location="cpu")
            step = int(payload.get("global_step", -1))
        mapping[step] = file
    return mapping


def load_state_vector(snapshot_path: Path, component: str) -> torch.Tensor:
    payload = torch.load(snapshot_path, map_location="cpu")
    state_dict = payload["state_dict"]
    if component == "text":
        items = [tensor.flatten() for name, tensor in state_dict.items() if name.startswith("text_model")]
    elif component == "vision":
        items = [tensor.flatten() for name, tensor in state_dict.items() if name.startswith("vision_model")]
    else:
        items = [tensor.flatten() for tensor in state_dict.values()]
    if not items:
        raise RuntimeError(f"{snapshot_path} 中未找到 {component} 分支的 LoRA 权重")
    return torch.cat(items)


def compute_similarity(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    vec_a = vec_a / vec_a.norm(p=2)
    vec_b = vec_b / vec_b.norm(p=2)
    return torch.dot(vec_a, vec_b).item()


def ensure_output_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    snaps_a = list_snapshots(args.client_a)
    snaps_b = list_snapshots(args.client_b)

    common_steps = sorted(set(snaps_a.keys()) & set(snaps_b.keys()))
    if not common_steps:
        raise RuntimeError("两个客户端没有对齐的快照 step，请检查记录频率是否一致。")

    rows: List[Tuple[int, float, float]] = []
    for step in common_steps:
        vec_text_a = load_state_vector(snaps_a[step], "text")
        vec_text_b = load_state_vector(snaps_b[step], "text")
        vec_vis_a = load_state_vector(snaps_a[step], "vision")
        vec_vis_b = load_state_vector(snaps_b[step], "vision")

        text_cos = compute_similarity(vec_text_a, vec_text_b)
        vision_cos = compute_similarity(vec_vis_a, vec_vis_b)
        rows.append((step, text_cos, vision_cos))

    ensure_output_dir(args.output_csv)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "text_cos", "vision_cos"])
        writer.writerows(rows)

    ensure_output_dir(args.output_fig)
    steps = [r[0] for r in rows]
    text_sim = [r[1] for r in rows]
    vision_sim = [r[2] for r in rows]

    plt.figure(figsize=(8, 4))
    plt.plot(steps, text_sim, label="Text LoRA Cosine", marker="o")
    plt.plot(steps, vision_sim, label="Vision LoRA Cosine", marker="x")
    plt.xlabel("Step")
    plt.ylabel("Cosine Similarity")
    plt.title("Client LoRA Similarity Over Training")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output_fig, dpi=200)
    print(f"已保存 CSV 至 {args.output_csv}，图像至 {args.output_fig}")


if __name__ == "__main__":
    main()

