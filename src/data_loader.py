# src/data_loader.py (最终修正版 - 含 Label Shuffle 攻击)
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import random # 必须导入

import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
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
        self.root = Path(root)
        self.domain = domain
        self.transform = transform

        # 尝试查找目录，兼容带空格或下划线的情况
        domain_dir = self.root / domain
        if not domain_dir.exists():
            domain_dir = self.root / domain.replace(" ", "_")
            if not domain_dir.exists():
                 raise FileNotFoundError(f"域目录 {domain}/{domain.replace(' ', '_')} 不存在于 {self.root}")

        self.samples: List[Tuple[Path, int]] = []
        self.class_to_idx: Dict[str, int] = {}

        # 遍历类别文件夹
        valid_class_dirs = sorted([p for p in domain_dir.iterdir() if p.is_dir()])
        
        for class_idx, class_dir in enumerate(valid_class_dirs):
            self.class_to_idx[class_dir.name] = class_idx
            for img_path in class_dir.glob("**/*"):
                if img_path.suffix.lower() in ALLOWED_EXTS:
                    self.samples.append((img_path, class_idx))

        if not self.samples:
            raise RuntimeError(f"{domain_dir} 下未找到图像")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        # 这里的 samples 列表如果被修改了，读取到的标签就会变
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
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_IMAGE_MEAN, CLIP_IMAGE_STD),
        ]
    )


def create_officehome_dataloader(
    root: str,
    domain: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    transform: Optional[Callable] = None,
    data_ratio: float = 1.0,
    malicious: bool = False  # <--- [修复] 加回 malicious 参数
) -> Tuple[DataLoader, DatasetInfo]:
    """
    构建 DataLoader。
    """
    if transform is None:
        transform = build_train_transform()

    # 1. 实例化完整数据集
    dataset = OfficeHomeDataset(root=root, domain=domain, transform=transform)
    
    # === [关键修复] 植入造毒逻辑 (Label Shuffle) ===
    # 必须在 Subset 之前执行，直接修改 dataset.samples
    if malicious:
        print(f"!!! [ATTACK] Applying Label Shuffle on Domain: {domain} !!!")
        
        # 1. 创建映射表
        num_classes = len(dataset.class_to_idx)
        # 固定种子，保证每次运行毒药一致
        g = torch.Generator()
        g.manual_seed(42) 
        perm = torch.randperm(num_classes, generator=g).tolist()
        label_map = {i: perm[i] for i in range(num_classes)}
        
        print(f"    > Label Map Example: 0->{label_map[0]}, 1->{label_map[1]}...")

        # 2. 修改 dataset.samples
        # OfficeHomeDataset 的 samples 是 [(path, label), ...]
        new_samples = []
        for path, target in dataset.samples:
            new_target = label_map[target]
            new_samples.append((path, new_target))
        
        dataset.samples = new_samples
        print(f"!!! [ATTACK] Successfully corrupted {len(dataset)} samples !!!")
    
    # 2. 处理数据子采样 (Scenario C)
    if data_ratio < 1.0:
        total_len = len(dataset)
        keep_len = int(total_len * data_ratio)
        keep_len = max(keep_len, batch_size) 
        
        if keep_len < total_len:
            indices = np.random.choice(total_len, keep_len, replace=False)
            dataset = Subset(dataset, indices)
            print(f"[{domain}] Data Subsampling Enabled: {total_len} -> {keep_len} samples ({data_ratio:.2f})")
    
    # 3. 构建 Loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    # 处理 Subset 的 info 获取
    base_ds = dataset.dataset if isinstance(dataset, Subset) else dataset
    
    info = base_ds.info()
    info.num_samples = len(dataset)
    
    return dataloader, info