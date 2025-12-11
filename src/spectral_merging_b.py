# spectral_merging_b.py
# B 版算法：Pruning as Alignment（基于公共锚点的敏感度剪枝）

from __future__ import annotations

import copy
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import swanlab


class SensitivityAggregator:
    """
    基于梯度敏感度的语义剪枝聚合器 (Pruning as Alignment)。

    这里只负责“如何用给定的 model + anchor_dataloader 做剪枝和聚合”，
    不负责构建 CLIP+LoRA 模型或锚点 DataLoader，这些在 server/strategy 层完成。
    """

    def __init__(
        self,
        model,
        anchor_dataloader,
        device: torch.device | str = "cuda",
        prune_ratio: float = 0.7,
        client_domains: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            model: 基座模型 (CLIP + LoRA 结构)，需与客户端训练时结构完全一致。
            anchor_dataloader: 公共锚点 DataLoader，batch 形如：
                               {"pixel_values": Tensor[B,3,H,W], "input_ids": Tensor[B,L]}
            device: 计算设备。
            prune_ratio: 剪枝率，0.7 表示认为 70% 的低敏感度参数是噪音，会被裁掉。
            client_domains: 客户端域名列表，用于在 SwanLab 里打 Server/Anchor_Loss_{Domain} 等指标。
        """
        self.model = model
        self.anchor_dataloader = anchor_dataloader
        self.device = device
        self.prune_ratio = prune_ratio
        self.client_domains: Optional[List[str]] = client_domains

        # aggregate 被调用的轮次计数，用作 SwanLab 中的 round 维度
        self.round_index: int = 0

    def compute_saliency_and_prune(
        self,
        client_state_dict: Dict[str, torch.Tensor],
        client_index: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        对单个客户端 LoRA 参数做【体检 -> 剪枝 -> 缩放】。

        输入：只包含 LoRA adapter 权重的 state_dict
        输出：剪枝 + 能量补偿之后的 LoRA state_dict
        """
        # 1. 加载客户端 LoRA 权重到模型（strict=False，因为 state_dict 只包含 LoRA 部分）
        self.model.load_state_dict(client_state_dict, strict=False)
        self.model.to(self.device)

        # 2. 只对 LoRA 参数开启梯度，冻结基座权重
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.model.zero_grad()

        # 3. 前向 + 反向 (Diagnosis)：在公共锚点上计算 CLIP 对比损失，产生梯度
        total_loss = 0.0
        batch_count = 0

        for batch in self.anchor_dataloader:
            if not isinstance(batch, dict):
                raise TypeError(
                    "anchor_dataloader must return a dict with keys "
                    "'pixel_values' and 'input_ids'."
                )

            images = batch["pixel_values"].to(self.device, non_blocking=True)
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)

            outputs = self.model(input_ids=input_ids, pixel_values=images)
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text

            batch_size = images.size(0)
            targets = torch.arange(batch_size, device=self.device)

            loss_i = F.cross_entropy(logits_per_image, targets)
            loss_t = F.cross_entropy(logits_per_text, targets)
            loss = (loss_i + loss_t) / 2.0

            loss.backward()

            total_loss += float(loss.item())
            batch_count += 1

            # 诊断只需要少量 batch，减少显存与时间开销
            if batch_count >= 5:
                break

        anchor_loss_avg: Optional[float] = None
        if batch_count > 0:
            anchor_loss_avg = total_loss / batch_count
            print(f"    > Anchor Loss: {anchor_loss_avg:.4f}")

        # SwanLab：构造当前客户端的展示名称
        client_name: Optional[str] = None
        if client_index is not None:
            if self.client_domains and 0 <= client_index < len(self.client_domains):
                client_name = self.client_domains[client_index]
            else:
                client_name = f"Client_{client_index}"

        # 3.1 SwanLab：记录 Server/Anchor_Loss_{Domain}
        if anchor_loss_avg is not None and client_name is not None:
            try:
                swanlab.log(
                    {
                        "round": self.round_index,
                        f"Server/Anchor_Loss/{client_name}": float(anchor_loss_avg),
                    }
                )
            except Exception:
                # 不影响算法本身执行
                pass

        # 4. 基于梯度的剪枝 + 能量补偿 (Surgery)
        processed_state_dict: Dict[str, torch.Tensor] = {}
        saliency_means: List[float] = []

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if "lora_" not in name:
                    continue

                if param.grad is None:
                    # 若完全无梯度，视为“死参数”
                    processed_state_dict[name] = torch.zeros_like(param.data)
                    continue

                # --- Step A: 一阶泰勒展开 Score = |W * ∇W| ---
                saliency = (param.data * param.grad).abs()
                saliency_means.append(float(saliency.mean().item()))

                # --- Step B: 按 prune_ratio 找阈值并剪枝 ---
                num_params = saliency.numel()
                if num_params == 0:
                    processed_state_dict[name] = param.data
                    continue

                k = int(num_params * self.prune_ratio)
                if k > 0:
                    threshold = torch.kthvalue(saliency.view(-1), k).values
                    mask = (saliency >= threshold).float()
                else:
                    mask = torch.ones_like(saliency)

                pruned_weight = param.data * mask

                # --- Step C: 能量补偿缩放 ---
                energy_original = param.data.abs().sum()
                energy_pruned = pruned_weight.abs().sum()

                if energy_pruned > 1e-6:
                    scale_factor = energy_original / energy_pruned
                    scale_factor = torch.clamp(scale_factor, max=10.0)
                else:
                    scale_factor = 1.0

                processed_state_dict[name] = pruned_weight * scale_factor

        # 4.1 SwanLab：整客户端的敏感度指标 Server/Saliency_Mean_{Domain}
        if saliency_means and client_name is not None:
            saliency_mean_overall = float(sum(saliency_means) / len(saliency_means))
            try:
                swanlab.log(
                    {
                        "round": self.round_index,
                        f"Server/Saliency_Mean/{client_name}": saliency_mean_overall,
                    }
                )
            except Exception:
                pass

        return processed_state_dict

    def aggregate(self, client_state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        对所有客户端做剪枝 + 缩放后，再做简单平均 (AvgMerge)。
        """
        n_clients = len(client_state_dicts)
        processed_models: List[Dict[str, torch.Tensor]] = []

        # 为 SwanLab 增加 round 维度（与 FedServer 的通信轮次对齐）
        self.round_index += 1

        print(f"Starting Sensitivity-Based Pruning (Ratio={self.prune_ratio})...")

        for idx, client_dict in enumerate(client_state_dicts):
            print(f"  > Processing Client {idx} ...")
            processed = self.compute_saliency_and_prune(client_dict, client_index=idx)
            processed_models.append(processed)

        print("  > Aggregating processed models...")
        avg_state_dict: Dict[str, torch.Tensor] = copy.deepcopy(processed_models[0])

        for key, value in avg_state_dict.items():
            if not isinstance(value, torch.Tensor):
                continue

            summed = processed_models[0][key].clone()
            for i in range(1, n_clients):
                summed += processed_models[i][key]
            avg_state_dict[key] = summed / float(n_clients)

        return avg_state_dict

