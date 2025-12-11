# 本文件封装 CLIP+LoRA 的训练逻辑与权重快照导出
# 每个客户端都会实例化一个 ClipLoRALearner 完成本地微调
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from transformers import AutoTokenizer, CLIPModel

PROMPT_TEMPLATE = "A photo of a {}."


@dataclass
class LearnerConfig:
    model_name: str = "openai/clip-vit-base-patch32"
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    logging_steps: int = 50
    snapshot_steps: int = 100
    fp16: bool = True


class ClipLoRALearner:
    """封装 CLIP LoRA 本地训练流程，并支持恶意客户端注入。"""

    def __init__(
        self,
        class_names: Sequence[str],
        device: torch.device,
        lora_cfg: Dict,
        learner_cfg: Optional[LearnerConfig] = None,
        output_dir: str = "checkpoints",
        is_malicious: bool = False,
        poison_shuffle_prob: float = 1.0,
    ) -> None:
        self.device = device
        self.class_names = list(class_names)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cfg = learner_cfg or LearnerConfig()
        self.is_malicious = is_malicious
        self.poison_shuffle_prob = poison_shuffle_prob

        self.model = CLIPModel.from_pretrained(self.cfg.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)

        self._attach_lora(lora_cfg)
        self.model.to(self.device)
        self.model.train()

        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        self.scaler = GradScaler(enabled=self.cfg.fp16)

        self.text_inputs = self._build_text_tokens()
        self.global_step = 0

        if self.is_malicious:
            print("⚠️ 警告：该客户端启用了语义攻击（标签洗牌）。")

    def _attach_lora(self, lora_cfg: Dict) -> None:
        lora_config = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            bias="none",
            target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj"]),
            task_type="FEATURE_EXTRACTION",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def _build_text_tokens(self):
        prompts = [PROMPT_TEMPLATE.format(name.replace("_", " ")) for name in self.class_names]
        tokens = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in tokens.items()}

    def _compute_logits(self, pixel_values: torch.Tensor):
        outputs = self.model(pixel_values=pixel_values, **self.text_inputs)
        return outputs.logits_per_image

    def train_step(self, batch):
        """执行一次前向+反向，可选语义攻击。"""
        images, labels = batch
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        if self.is_malicious and torch.rand(1).item() < self.poison_shuffle_prob:
            perm = torch.randperm(labels.size(0), device=labels.device)
            labels = labels[perm]

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=self.cfg.fp16):
            logits = self._compute_logits(images)
            loss = F.cross_entropy(logits, labels)
        self.scaler.scale(loss).backward()
        if self.cfg.grad_clip is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        preds = torch.argmax(logits, dim=-1)
        acc = (preds == labels).float().mean().item()
        self.global_step += 1
        return {"loss": loss.item(), "acc": acc}

    def save_checkpoint(self, output_name: str, extra: Optional[Dict] = None) -> str:
        path = self.output_dir / output_name
        payload = {
            "global_step": self.global_step,
            "model_name": self.cfg.model_name,
            "class_names": self.class_names,
            "state_dict": self.model.get_peft_model_state_dict(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        return str(path)

    def save_lora_snapshot(self, tag: str) -> str:
        """仅保存 LoRA ΔW，供相似度分析使用。"""
        state = self.model.get_peft_model_state_dict()
        path = self.output_dir / f"lora_{tag}.pt"
        torch.save({"global_step": self.global_step, "state_dict": state}, path)
        return str(path)

    def extract_lora_delta(self, component: str = "all") -> Dict[str, torch.Tensor]:
        """返回指定模块的 LoRA 权重，component∈{all,text,vision}。"""
        valid = {"all", "text", "vision"}
        if component not in valid:
            raise ValueError(f"component 必须是 {valid}")

        state = self.model.get_peft_model_state_dict()
        if component == "all":
            return state

        keyword = "text_model" if component == "text" else "vision_model"
        return {k: v for k, v in state.items() if keyword in k}

    def load_lora_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """加载仅包含 LoRA Adapter 的参数，用于衔接联邦聚合。"""
        set_peft_model_state_dict(self.model, state_dict, adapter_name="default")

    def set_device(self, device: torch.device) -> None:
        """在多进程环境下重新绑定 GPU。"""
        self.device = device
        if device.type == "cuda":
            torch.cuda.set_device(device)
        self.model.to(device)
        self.text_inputs = {k: v.to(device) for k, v in self.text_inputs.items()}
