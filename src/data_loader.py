# 本文件负责加载 Office-Home 数据并构建 PyTorch DataLoader
# 通过域划分模拟联邦客户端的视觉发散场景
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
OFFICEHOME_DOMAINS = ["Art", "Clipart", "Product", "Real_World"]


@dataclass
class DatasetInfo:
    domain: str
    num_classes: int
    num_samples: int
    class_to_idx: Dict[str, int]


class OfficeHomeDataset(Dataset):
    """按域划分的 Office-Home 数据集。"""

    def __init__(
        self,
        root: str,
        domain: str,
        transform: Optional[Callable] = None,
    ) -> None:
        normalized_domain = domain
        if normalized_domain not in OFFICEHOME_DOMAINS:
            raise ValueError(f"不支持的域 {domain}，请从 {OFFICEHOME_DOMAINS} 中选择")

        # 数据根目录来自 configs/local.yaml 中的 data.office_home_root
        # 例如：/data1/lc/data/office_home（与代码仓库路径无关）
        self.root = Path(root)
        self.domain = normalized_domain
        self.transform = transform

        domain_dir = self.root / normalized_domain
        if not domain_dir.exists():
            raise FileNotFoundError(f"域目录 {domain_dir} 不存在，请先下载 Office-Home 数据")

        self.samples: List[Tuple[Path, int]] = []
        self.class_to_idx: Dict[str, int] = {}

        for class_idx, class_dir in enumerate(sorted(p for p in domain_dir.iterdir() if p.is_dir())):
            self.class_to_idx[class_dir.name] = class_idx
            for img_path in class_dir.glob("**/*"):
                if img_path.suffix.lower() in ALLOWED_EXTS:
                    self.samples.append((img_path, class_idx))

        if not self.samples:
            raise RuntimeError(f"{domain_dir} 下未找到支持的图像文件格式：{ALLOWED_EXTS}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def info(self) -> DatasetInfo:
        return DatasetInfo(
            domain=self.domain,
            num_classes=len(self.class_to_idx),
            num_samples=len(self.samples),
            class_to_idx=self.class_to_idx,
        )


def build_train_transform(image_size: int = 224) -> Callable:
    """CLIP 风格的数据增强。"""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_IMAGE_MEAN, CLIP_IMAGE_STD),
        ]
    )


def build_eval_transform(image_size: int = 224) -> Callable:
    """服务器侧评估时使用的裁剪策略。"""
    return transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_IMAGE_MEAN, CLIP_IMAGE_STD),
        ]
    )


def create_officehome_dataloader(
    root: str,
    domain: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    transform: Optional[Callable] = None,
    data_ratio: float = 1.0 # 数据比例，如果小于 1.0，则随机保留该比例的数据（用于模拟 Few-Shot/Imbalance）
) -> Tuple[DataLoader, DatasetInfo]:
    """构建 DataLoader 并返回数据集元信息。"""
    if transform is None:
        transform = build_train_transform()

    dataset = OfficeHomeDataset(root=root, domain=domain, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return dataloader, dataset.info()


def export_class_mapping(info: DatasetInfo, output_path: str) -> None:
    """将类别映射写出，便于日志或可视化引用。"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump({"domain": info.domain, "class_to_idx": info.class_to_idx}, f, ensure_ascii=False, indent=2)


def batch_to_device(batch, device: torch.device):
    """便捷地把一个 batch 搬到目标 GPU。"""
    images, labels = batch
    return images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

