# eval_cifar100.py
import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPModel

# 允许从任意工作目录运行
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data_loader import CLIP_IMAGE_MEAN, CLIP_IMAGE_STD

#定义数据目录为相对路径
project_root = Path(__file__).resolve().parent
default_data_root = (project_root / "../../data/CIFAR100").resolve()

PROMPT_TEMPLATE = "a photo of a {}."

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_lora_state_dict(checkpoint_path: str) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and isinstance(ckpt.get("state_dict"), dict):
        return ckpt["state_dict"]
    if isinstance(ckpt, dict) and isinstance(ckpt.get("model_state_dict"), dict):
        return ckpt["model_state_dict"]
    return ckpt

def build_eval_transform(image_size: int = 224):
    resize_interp = transforms.InterpolationMode.BICUBIC
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=resize_interp),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_IMAGE_MEAN, CLIP_IMAGE_STD),
    ])

@torch.no_grad()
def eval_zeroshot(model, tokenizer, loader, class_names, device):
    model.eval()

    prompts = [PROMPT_TEMPLATE.format(n.replace("_", " ")) for n in class_names]
    tok = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    tok = {k: v.to(device) for k, v in tok.items()}

    text_feat = model.get_text_features(**tok)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="CIFAR-100"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        img_feat = model.get_image_features(pixel_values=images)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        logits = img_feat @ text_feat.t()
        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
    return correct / total

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="src/configs/fed_ours_b.yaml")
    p.add_argument("--checkpoint", default=None, help="不传则评 base CLIP")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data_root", type=str, default=str(default_data_root))

    args = p.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg.get("train", {})
    lora_cfg = cfg.get("lora", {})
    model_name = train_cfg.get("model_name", "openai/clip-vit-base-patch32")

    device = torch.device(args.device)
    base = CLIPModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if args.checkpoint:
        peft_cfg = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            bias=lora_cfg.get("bias", "none"),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj"]),
        )
        model = get_peft_model(base, peft_cfg)
        lora_sd = load_lora_state_dict(args.checkpoint)
        set_peft_model_state_dict(model, lora_sd, adapter_name="default")
    else:
        model = base

    model.to(device)

    ds = datasets.CIFAR100(root=args.data_root, train=False, download=True, transform=build_eval_transform())
    class_names = list(ds.classes)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    acc = eval_zeroshot(model, tokenizer, loader, class_names, device)
    print(f"CIFAR-100 zero-shot top1: {acc:.2%}")

if __name__ == "__main__":
    main()
