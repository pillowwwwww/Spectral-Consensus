from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from src.data_loader import build_train_transform


NumImages = Union[int, Tuple[int, int]]

# =========================================================
# 你要的相对路径默认值（从 src/ 下的 dataloader 文件位置出发）
# =========================================================
image_root = "../../data/COCO2017/val2017"
captions_json = "../../data/COCO2017/annotations/captions_val2017.json"


class CocoAnchorDataset(Dataset):
    """
    COCO 公共锚点数据集：

    功能：
    - 从 COCO captions 标注文件中随机抽取若干张图片
    - 每张图随机挑选一条 caption
    - 构造 CLIP 风格的 (image, text) 对

    关键改动：
    - num_images 支持：
        * int: 固定数量
        * (min, max): 在 [min, max] 区间内随机采样数量
    - 抽样逻辑改为：先按 image_id 聚合 caption，再随机选 image_id
    """

    def __init__(
        self,
        image_root: str,
        captions_json: str,
        tokenizer,
        num_images: NumImages = (32, 64),
        transform=None,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__()
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.transform = transform or build_train_transform()

        # 使用局部 RNG，避免污染全局 random 状态
        rng = random.Random(seed)

        with open(captions_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        images: Dict[int, str] = {img["id"]: img["file_name"] for img in data["images"]}
        annotations: List[Dict] = list(data["annotations"])

        # image_id -> captions
        caps_by_image: Dict[int, List[str]] = {}
        for ann in annotations:
            image_id = ann.get("image_id")
            caption = (ann.get("caption") or "").strip()
            if not image_id or not caption:
                continue
            caps_by_image.setdefault(image_id, []).append(caption)

        # 计算最终抽取数量
        if isinstance(num_images, tuple):
            min_n, max_n = num_images
            if min_n <= 0 or max_n <= 0 or min_n > max_n:
                raise ValueError(f"num_images tuple invalid: {num_images}")
            target_n = rng.randint(min_n, max_n)
        else:
            if num_images <= 0:
                raise ValueError(f"num_images must be positive, got {num_images}")
            target_n = num_images

        # 候选 image_ids（必须同时满足：有 file_name + 有 caption + 文件存在）
        candidate_ids: List[int] = []
        for image_id in caps_by_image.keys():
            file_name = images.get(image_id)
            if not file_name:
                continue
            img_path = os.path.join(self.image_root, file_name)
            if os.path.isfile(img_path):
                candidate_ids.append(image_id)

        if not candidate_ids:
            raise RuntimeError(
                f"CocoAnchorDataset: no valid candidates found under {image_root}"
            )

        # 如果候选不足，自动缩小 target
        target_n = min(target_n, len(candidate_ids))

        # 采样 image_id
        picked_ids = rng.sample(candidate_ids, k=target_n)

        samples: List[Dict] = []
        for image_id in picked_ids:
            file_name = images[image_id]
            img_path = os.path.join(self.image_root, file_name)
            captions = caps_by_image[image_id]
            caption = rng.choice(captions)
            samples.append({"image_path": img_path, "caption": caption})

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.samples[idx]
        image_path: str = record["image_path"]
        caption: str = record["caption"]

        with Image.open(image_path) as img:
            img = img.convert("RGB")

        img_tensor = self.transform(img) if self.transform is not None else img

        # 让 max_length 更可控（不依赖 tokenizer 隐式默认也行）
        max_len = getattr(self.tokenizer, "model_max_length", None)
        tokens = self.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_len if isinstance(max_len, int) and max_len > 0 else None,
        )
        input_ids = tokens["input_ids"].squeeze(0)

        return {
            "pixel_values": img_tensor,
            "input_ids": input_ids,
        }


def build_coco_anchor_dataloader(
    tokenizer,
    image_root: str = image_root,
    captions_json: str = captions_json,
    batch_size: int = 32,
    num_images: NumImages = (32, 64),
    num_workers: int = 4,
    seed: Optional[int] = 42,
    shuffle: bool = False,
) -> DataLoader:
    """
    构建 COCO 公共锚点 DataLoader。

    Args:
        tokenizer: 与 CLIP 主模型一致的 tokenizer
        image_root: COCO 图片根目录（默认使用相对路径）
        captions_json: COCO captions 标注文件路径（默认使用相对路径）
        batch_size: DataLoader batch 大小
        num_images:
            - int: 固定抽取数量
            - (min, max): 在区间内随机抽取数量
        num_workers: DataLoader 的 num_workers
        seed: 随机种子，控制抽样可复现
        shuffle: 是否打乱 DataLoader 内部顺序（默认 False，保证锚点顺序更稳定）

    Returns:
        DataLoader，其每个 batch 是：
            {"pixel_values": Tensor[B,3,H,W], "input_ids": Tensor[B,L]}
    """
    dataset = CocoAnchorDataset(
        image_root=image_root,
        captions_json=captions_json,
        tokenizer=tokenizer,
        num_images=num_images,
        transform=None,
        seed=seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader
