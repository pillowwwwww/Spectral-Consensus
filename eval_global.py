import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPModel
import swanlab
# 确保可以导入 src 包
sys.path.append(str(Path(__file__).resolve().parent / "src"))
from src.data_loader import create_officehome_dataloader
from src.learner import PROMPT_TEMPLATE


def load_config(path: str):
    if not os.path.exists(path):
        print(f"Error: 配置文件 {path} 不存在。")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_text_inputs(tokenizer: AutoTokenizer, class_names, device: str):
    """构造与训练阶段一致的文本提示 tokens。"""
    prompts = [PROMPT_TEMPLATE.format(name.replace("_", " ")) for name in class_names]
    tokens = tokenizer(prompts, padding=True, return_tensors="pt")
    return {k: v.to(device) for k, v in tokens.items()}


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    text_inputs: dict,
) -> float:
    """在给定 DataLoader 上评估分类准确率（使用 CLIP image-text 头）。"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            # data_loader 返回的是 (images, labels)
            if len(batch) == 2:
                images, labels = batch
            elif len(batch) >= 3:
                images, labels = batch[0], batch[1]
            else:
                raise ValueError("DataLoader 返回格式无法解析")

            images = images.to(device)
            labels = labels.to(device)

            # 与训练阶段一致：image + text 共同前向，使用 logits_per_image
            outputs = model(pixel_values=images, **text_inputs)
            logits = outputs.logits_per_image

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        return 0.0
    return correct / total


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Global CLIP+LoRA Model on Office-Home Domains"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/fed_ours_b.yaml",
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="全局模型权重文件 (.pt) 路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="计算设备",
    )
    args = parser.parse_args()

    # 1. 加载配置
    print(f"Loading config from {args.config}...")
    cfg = load_config(args.config)
    # ==== SwanLab: 初始化一个 eval 任务 ====
    train_cfg = cfg.get("train", {})
    lora_cfg = cfg.get("lora", {})

    swanlab.init(
        project="myfedclip",   # 和训练时保持一致
        workspace="chang",
        config={
            "mode": "eval",
            "config_path": args.config,
            "checkpoint": args.checkpoint,
            "strategy": train_cfg.get("strategy", "fedavg"),
            "rounds": train_cfg.get("rounds"),
            "epochs_per_round": train_cfg.get("epochs_per_round"),
            "model_name": train_cfg.get("model_name"),
            "batch_size": train_cfg.get("batch_size"),
            "learning_rate": train_cfg.get("learning_rate"),
            "lora_r": lora_cfg.get("r"),
        },
    )
    # ==== SwanLab init 结束 ====
    # 2. 准备基础 CLIP 模型
    model_name = cfg.get("train", {}).get(
        "model_name", "openai/clip-vit-base-patch32"
    )
    print(f"Loading base model: {model_name}...")
    base_model = CLIPModel.from_pretrained(model_name)

    # 3. 配置 LoRA（必须与训练时一致）
    lora_cfg = cfg.get("lora", {})
    print(f"Applying LoRA config: {lora_cfg}")

    peft_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        target_modules=lora_cfg.get(
            "target_modules", ["q_proj", "k_proj", "v_proj"]
        ),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
    )

    model = PeftModel(base_model, peft_config)

    # 4. 加载全局 LoRA checkpoint（只包含 LoRA 参数）
    print(f"Loading checkpoint weights from {args.checkpoint}...")
    try:
        state_dict = torch.load(args.checkpoint, map_location=args.device)
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Load result: {msg}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    model.to(args.device)

    # 文本 tokenizer，与训练保持一致
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 5. 为四个域构建测试集 DataLoader（全部 shuffle=False）
    domains = ["Art", "Clipart", "Product", "Real_World"]
    data_root = cfg.get("data", {}).get(
        "office_home_root", "./data/OfficeHome"
    )

    results = {}
    print("\n" + "=" * 40)
    print(f"{'Domain':<15} | {'Accuracy':<10}")
    print("-" * 28)

    for domain in domains:
        try:
            # create_officehome_dataloader 返回 (dataloader, DatasetInfo)
            test_loader, dataset_info = create_officehome_dataloader(
                root=data_root,
                domain=domain,
                batch_size=32,
                num_workers=4,
                shuffle=False,  # 评估阶段：只打乱 label=False，顺序固定
            )
            class_names = sorted(
                dataset_info.class_to_idx,
                key=dataset_info.class_to_idx.get,
            )
        except Exception as e:
            print(f"Error creating dataloader for {domain}: {e}")
            continue

        text_inputs = build_text_inputs(tokenizer, class_names, args.device)
        acc = evaluate(model, test_loader, args.device, text_inputs)
        results[domain] = acc
        print(f"{domain:<15} | {acc:.2%}")
        
        # ==== SwanLab: 记录每个域的精度 ====
        try:
            swanlab.log({f"Eval/Accuracy/{domain}": float(acc)})
        except Exception:
            pass
        # ==== SwanLab log 结束 ====

    print("=" * 40)
    if results:
        avg_acc = sum(results.values()) / len(results)
        print(f"{'Average':<15} | {avg_acc:.2%}")
                # ==== SwanLab: 记录平均精度 ====
        try:
            swanlab.log({"Eval/Accuracy/avg_domains": float(avg_acc)})
        except Exception:
            pass
        # ==== SwanLab avg log 结束 ====
    else:
        print("No results collected.")

    # ==== SwanLab: 结束本次 eval 任务 ====
    swanlab.finish()

    
if __name__ == "__main__":
    main()

