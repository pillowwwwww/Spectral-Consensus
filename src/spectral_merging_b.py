# spectral_merging_b.py
# B 版算法：Pruning as Alignment（基于公共锚点的敏感度剪枝）

from __future__ import annotations

import copy
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import swanlab
from peft import set_peft_model_state_dict, get_peft_model_state_dict
from transformers import CLIPTokenizer
import torch.nn.functional as F
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

    def compute_saliency_and_prune(self, client_state_dict: Dict[str, torch.Tensor], client_index: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        [终极修复版] 核心方法：对单个客户端的 LoRA 参数进行【体检 -> 剪枝 -> 缩放】
        
        功能清单：
        1. ✅ 使用官方 API (set_peft_model_state_dict) 解决 Key Mismatch。
        2. ✅ 增加 B 矩阵非零检查，防止加载空壳参数。
        3. ✅ 自动加载 Tokenizer (支持 HF 镜像/本地缓存)。
        4. ✅ 鲁棒的数据解包：兼容 PyTorch List/Tuple 和 HuggingFace Dict。
        5. ✅ 智能文本构造：优先用真实标签 (dataset.classes)，失败则用 Dummy Prompt。
        """
        print(f"\n ✅ [Server] 开始处理客户端 {client_index} 的参数 (Saliency Pruning)...")
        
        # =======================================================
        # 1. 加载参数 (Loading with Official API)
        # =======================================================
        try:
            # 官方 API 会自动处理 base_model.model 前缀问题
            set_peft_model_state_dict(self.model, client_state_dict)
        except Exception as e:
            print(f"❌ [加载异常] set_peft_model_state_dict 抛出错误: {e}")
            raise e

        self.model.to(self.device)
        
        # =======================================================
        # 🛡️ 防御层: 验证 LoRA 是否真的加载进去了？
        # =======================================================
        zero_b_count = 0
        total_b_count = 0
        for name, param in self.model.named_parameters():
            if "lora_B" in name:
                total_b_count += 1
                if torch.all(param.data == 0):
                    zero_b_count += 1
        
        if total_b_count > 0 and zero_b_count == total_b_count:
            raise RuntimeError("❌ [致命错误] Server 端 LoRA 参数加载失败！所有的 lora_B 矩阵都是 0！")
        elif zero_b_count > 0:
            print(f"⚠️ [警告] 发现 {zero_b_count}/{total_b_count} 个 lora_B 矩阵依然为 0。")
        else:
            print(f"✅ [成功] LoRA 参数加载验证通过 (B矩阵非零)。")

        # =======================================================
        # 2. 准备 Tokenizer & 真实标签映射
        # =======================================================
        tokenizer = None
        if CLIPTokenizer is not None:
            try:
                # 在终端配置 export HF_ENDPOINT=https://hf-mirror.com
                tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            except Exception as e:
                print(f"⚠️ [警告] Tokenizer 加载失败 ({e})，将尝试仅使用图像特征或跳过。")
        
        # 尝试从 DataLoader 提取真实的类别名称 (例如 ["Dog", "Cat", ...])
        real_class_names = None
        if hasattr(self.anchor_dataloader, 'dataset'):
            ds = self.anchor_dataloader.dataset
            if hasattr(ds, 'classes') and isinstance(ds.classes, (list, tuple)):
                real_class_names = ds.classes
                # print(f"[Server] 已提取真实类别表，共 {len(real_class_names)} 类")

        # =======================================================
        # 3. 准备梯度计算
        # =======================================================
        # 冻结非 LoRA 参数，开启 LoRA 梯度
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        self.model.zero_grad()
        
        # =======================================================
        # 4. 前向传播与反向传播 (Diagnosis)
        # =======================================================
        total_loss = 0.0
        batch_count = 0
        
        if len(self.anchor_dataloader) == 0:
             print("⚠️ [警告] Anchor DataLoader 为空！直接返回原参数。")
             return client_state_dict

        print(f"[Server] 正在使用公共锚点数据计算梯度敏感度...")
        
        for batch_idx, batch in enumerate(self.anchor_dataloader):
            images = None
            input_ids = None
            labels = None

            # --- [鲁棒解包] 兼容 Dict 和 List/Tuple ---
            if isinstance(batch, dict):
                images = batch.get('pixel_values')
                if images is None:
                    images = batch.get('images')
                input_ids = batch.get('input_ids') # 如果是 HF 处理好的数据，这里会有 input_ids
            elif isinstance(batch, (list, tuple)):
                images = batch[0]
                if len(batch) > 1:
                    # 检查第二个元素是 文本 还是 数字标签
                    second_element = batch[1]
                    if isinstance(second_element, torch.Tensor) and second_element.dtype in [torch.long, torch.int]:
                        labels = second_element # 是数字标签
                    else:
                        input_ids = second_element # 可能是 input_ids 或者 文本列表
            else:
                continue # 跳过未知格式

            if images is None:
                continue

            # --- [文本构造逻辑] ---
            # 优先级 1: DataLoader 直接提供了 input_ids -> 直接用
            # 优先级 2: 提供了 input_ids 文本列表 -> 现场 Tokenize
            # 优先级 3: 提供了数字标签 (labels) + 有对照表 (real_class_names) -> 查表造句 -> Tokenize
            # 优先级 4: 啥都没 -> 造假句 (Dummy) -> Tokenize

            if input_ids is None and tokenizer is not None:
                texts_to_tokenize = []
                
                # 尝试使用真实标签
                if labels is not None and real_class_names is not None:
                    class_indices = labels.tolist()
                    # 映射并清理下划线 (Alarm_Clock -> Alarm Clock)
                    names = [real_class_names[i].replace("_", " ") if i < len(real_class_names) else "object" for i in class_indices]
                    texts_to_tokenize = [f"a photo of a {name}" for name in names]
                    print(f"texts_to_tokenize 使用真实标签: {texts_to_tokenize}")
                # 否则使用兜底文本
                else:
                    texts_to_tokenize = ["a photo of an object"] * images.size(0)
                    print(f"texts_to_tokenize 使用兜底文本: {texts_to_tokenize}")
                # 执行 Tokenize
                try:
                    tokenized = tokenizer(texts_to_tokenize, padding=True, truncation=True, max_length=77, return_tensors="pt")
                    input_ids = tokenized["input_ids"]
                except Exception as e:
                    print(f"❌ Tokenize 失败: {e}")
                    continue

            # --- 再次检查 input_ids ---
            if input_ids is None:
                print("⚠️ [跳过] 无法构建文本输入，跳过此 Batch。")
                continue

            # 移动到 GPU
            images = images.to(self.device)
            input_ids = input_ids.to(self.device)
            
            # Forward
            # 注意：CLIP 需要 image 和 text 同时输入才能计算对比损失
            outputs = self.model(input_ids=input_ids, pixel_values=images)
            
            # Loss Calculation (Image-Text Matching)
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text
            
            # 构造对角线 Ground Truth (假设 Batch 内是一一对应的)
            current_bs = images.size(0)
            ground_truth = torch.arange(current_bs, device=self.device)
            
            loss = (F.cross_entropy(logits_per_image, ground_truth) + 
                    F.cross_entropy(logits_per_text, ground_truth)) / 2
            
            # Backward
            loss.backward()
            
            total_loss += loss.item()
            batch_count += 1
            
            # 只要跑 5 个 Batch 就够了
            if batch_count >= 5: 
                break
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        print(f"    > [Diagnosis] Anchor Loss: {avg_loss:.4f}")

        # =======================================================
        # 5. 剪枝与缩放 (Surgery)
        # =======================================================
        pruned_count = 0
        total_lora_params = 0
        # 用于统计敏感度的分布情况
        all_saliency_stats = []

        # 临时关闭梯度记录，进行 In-Place 修改
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    total_lora_params += 1
                    
                    if param.grad is None:
                        # 没有梯度的参数视为废弃，置零
                        param.data.fill_(0.0) 
                        continue
                    
                    # 计算敏感度
                    saliency = (param.data * param.grad).abs()
                    # --- [统计] 记录这一层的平均敏感度 ---
                    layer_mean = saliency.mean().item()
                    layer_max = saliency.max().item()
                    all_saliency_stats.append(layer_mean)

                    num_params = saliency.numel()
                    
                    if num_params > 0:
                        # 确定阈值
                        k = int(num_params * self.prune_ratio)
                        if k > 0:
                            threshold = torch.kthvalue(saliency.view(-1), k).values
                            mask = (saliency >= threshold).float()
                        else:
                            mask = torch.ones_like(saliency)
                        
                        # 保存原始能量用于缩放
                        original_data = param.data.clone()
                        
                        # 执行剪枝
                        param.data.mul_(mask)
                        # --- [统计] 这一层实际剪了多少 ---
                        # mask 里 0 的个数就是被剪掉的个数
                        layer_pruned = num_params - mask.sum().item()
                        pruned_count += int(layer_pruned)

                        
                        # 能量补偿 (Rescaling)
                        energy_original = original_data.abs().sum()
                        energy_pruned = param.data.abs().sum()
                        
                        if energy_pruned > 1e-6:
                            scale_factor = energy_original / energy_pruned
                            # 限制缩放倍数，防止数值爆炸
                            scale_factor = torch.clamp(scale_factor, max=10.0)
                            param.data.mul_(scale_factor)
                        
                        if k > 0: pruned_count += 1
        # =======================================================
        # 📊 [打印] 敏感度报告
        # =======================================================
        global_avg_saliency = sum(all_saliency_stats) / len(all_saliency_stats) if all_saliency_stats else 0
        prune_percentage = (pruned_count / total_lora_params * 100) if total_lora_params > 0 else 0
        
        print(f"    > [Report] 敏感度统计:")
        print(f"      - LoRA 参数总量: {total_lora_params}")
        print(f"      - 平均敏感度 (Mean Saliency): {global_avg_saliency:.6f} (如果不为0，说明计算成功)")
        print(f"      - 实际剪枝数量: {pruned_count} ({prune_percentage:.2f}%)")
        print(f"      - 目标剪枝率 (Ratio): {self.prune_ratio * 100}%")
        print(f"    > [Surgery] 完成剪枝。")

        # SwanLab 记录敏感度指标
        try:
            swanlab.log(
                {
                    "round": self.round_index,
                    "Server/Saliency/mean": float(global_avg_saliency),
                    "Server/Saliency/pruned_pct": float(prune_percentage),
                    "Server/Saliency/total_lora_params": int(total_lora_params),
                }
            )
        except Exception:
            pass

        # =======================================================
        # 6. 导出处理后的参数 (Export)
        # =======================================================
        # 使用官方 API 导出，确保 Key 格式标准，方便后续聚合
        final_dict = get_peft_model_state_dict(self.model)
        
        # 转回 CPU 节省显存
        final_dict = {k: v.cpu() for k, v in final_dict.items()}
            
        return final_dict
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
