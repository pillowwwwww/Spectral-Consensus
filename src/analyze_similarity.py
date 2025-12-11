from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# 适配你的文件名: "lora_art_step000050.pt" -> 提取出 50
SNAPSHOT_PATTERN = re.compile(r"(?:step)[-_]?(\d+)")

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze LoRA similarity")
    parser.add_argument("--client_a", type=str, required=True)
    parser.add_argument("--client_b", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="outputs/similarity.csv")
    parser.add_argument("--output_fig", type=str, default="outputs/similarity.png")
    # 只要 key 里包含这个字符串就算数
    parser.add_argument("--text_key", type=str, default="text_model")
    parser.add_argument("--vision_key", type=str, default="vision_model")
    return parser.parse_args()

def list_snapshots(directory: str) -> Dict[int, Path]:
    path = Path(directory)
    files = sorted([p for p in path.glob("*.pt")]) # 只看 .pt
    mapping: Dict[int, Path] = {}
    
    for file in files:
        # 忽略 final 这种非中间过程的文件，除非你想把 final 当作最后一步
        if "final" in file.name:
            continue

        match = SNAPSHOT_PATTERN.search(file.name)
        if match:
            step = int(match.group(1))
            mapping[step] = file
    return mapping

def extract_vectors(snapshot_path: Path, text_substring: str, vision_substring: str) -> Tuple[torch.Tensor, torch.Tensor]:
    print(f"Loading {snapshot_path.name}...", end="\r") # 进度条效果
    try:
        # 只加载 map_location="cpu"
        payload = torch.load(snapshot_path, map_location="cpu")
    except Exception as e:
        print(f"\nError loading {snapshot_path}: {e}")
        return torch.tensor([0.]), torch.tensor([0.])

    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload

    text_items = []
    vision_items = []

    for name, tensor in state_dict.items():
        flat_tensor = tensor.flatten().float()
        # 关键修改：从 startswith 改为 in (子字符串匹配)
        if text_substring in name:
            text_items.append(flat_tensor)
        elif vision_substring in name:
            vision_items.append(flat_tensor)

    if not text_items:
        print(f"\n[Alert] {snapshot_path.name}: 未找到包含 '{text_substring}' 的 Key！请检查 --text_key 参数。")
        vec_text = torch.tensor([0.])
    else:
        vec_text = torch.cat(text_items)

    if not vision_items:
        print(f"\n[Alert] {snapshot_path.name}: 未找到包含 '{vision_substring}' 的 Key！可能该模型没有视觉 LoRA。")
        vec_vis = torch.tensor([0.])
    else:
        vec_vis = torch.cat(vision_items)
        
    return vec_text, vec_vis

def compute_similarity(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    norm_a = vec_a.norm(p=2)
    norm_b = vec_b.norm(p=2)
    if norm_a < 1e-6 or norm_b < 1e-6:
        return 0.0
    return torch.dot(vec_a / norm_a, vec_b / norm_b).item()

def main():
    args = parse_args()
    snaps_a = list_snapshots(args.client_a)
    snaps_b = list_snapshots(args.client_b)

    common_steps = sorted(set(snaps_a.keys()) & set(snaps_b.keys()))
    print(f"Client A 路径: {args.client_a}")
    print(f"Client B 路径: {args.client_b}")
    
    if not common_steps:
        print("错误：两个文件夹没有找到 step 数字相同的 .pt 文件！")
        print(f"A steps: {list(snaps_a.keys())[:5]}...")
        print(f"B steps: {list(snaps_b.keys())[:5]}...")
        return

    print(f"找到 {len(common_steps)} 个对齐的 Step: {common_steps}")

    rows = []
    for step in common_steps:
        vec_text_a, vec_vis_a = extract_vectors(snaps_a[step], args.text_key, args.vision_key)
        vec_text_b, vec_vis_b = extract_vectors(snaps_b[step], args.text_key, args.vision_key)

        text_cos = compute_similarity(vec_text_a, vec_text_b)
        vision_cos = compute_similarity(vec_vis_a, vec_vis_b)
        rows.append((step, text_cos, vision_cos))

    # 输出
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "text_cos", "vision_cos"])
        writer.writerows(rows)

    # 绘图
    steps = [r[0] for r in rows]
    text_sim = [r[1] for r in rows]
    vision_sim = [r[2] for r in rows]

    steps_arr = np.array(steps)
    text_arr = np.array(text_sim)
    vision_arr = np.array(vision_sim)

    plt.figure(figsize=(10, 5))
    # 提高两条线的视觉区分度：颜色、线宽、标记大小
    plt.plot(
        steps_arr,
        text_arr,
        label="Text Similarity",
        color="#1f77b4",
        marker="o",
        markersize=5,
        linewidth=2.0,
    )
    plt.plot(
        steps_arr,
        vision_arr,
        label="Vision Similarity",
        color="#ff7f0e",
        marker="x",
        markersize=5,
        linewidth=2.0,
    )

    # 在两条曲线之间填充阴影区域，表示 Modality Gap
    upper = np.maximum(text_arr, vision_arr)
    lower = np.minimum(text_arr, vision_arr)
    plt.fill_between(
        steps_arr,
        lower,
        upper,
        alpha=0.18,
        color="gray",
        label="Modality Gap",
    )

    # 在中间位置添加 Modality Gap 文本标注
    mid_idx = len(steps_arr) // 2
    mid_x = steps_arr[mid_idx]
    mid_y = (text_arr[mid_idx] + vision_arr[mid_idx]) / 2.0
    plt.text(
        mid_x,
        mid_y,
        "Modality Gap",
        ha="center",
        va="center",
        fontsize=10,
        color="black",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, linewidth=0),
    )

    plt.xlabel("Training Step")
    plt.ylabel("Cosine Similarity")
    plt.title(f"LoRA Weight Similarity\n({Path(args.client_a).name} vs {Path(args.client_b).name})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output_fig, dpi=150)
    print(f"\nDone! CSV: {args.output_csv}, Image: {args.output_fig}")

if __name__ == "__main__":
    main()
