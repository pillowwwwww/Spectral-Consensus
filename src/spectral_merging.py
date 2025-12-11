import torch
import torch.nn.functional as F
import copy
from typing import List, Dict

class SpectralAggregator:
    def __init__(self, device='cuda', temperature=0.1):
        """
        Args:
            device: 计算设备
            temperature: Softmax 温度系数。越小，对低分客户端的惩罚越重（权重趋近于0）。
        """
        self.device = device
        self.temperature = temperature

    def merge_lora_to_matrix(self, state_dict: Dict, module_keyword: str) -> Dict[str, torch.Tensor]:
        """
        将 LoRA 的 A 和 B 矩阵相乘，恢复成 ΔW = B @ A。
        SVD 必须在完整矩阵上做才有物理意义（方向性）。
        
        Args:
            state_dict: 模型的参数字典
            module_keyword: "text_model" (锚点) 或 "vision_model" (被纠正对象)
        """
        delta_w_dict = {}
        
        # 遍历参数，寻找成对的 A/B 矩阵
        for key in state_dict.keys():
            # 筛选特定模态 (text 或 vision) 且是 LoRA A 矩阵的参数
            if module_keyword in key and "lora_A" in key:
                # 构造对应的 B 矩阵 key (Peft 命名规则通常是 lora_A.weight / lora_B.weight)
                key_A = key
                key_B = key.replace("lora_A", "lora_B")
                
                if key_B not in state_dict:
                    continue

                # 获取权重并转为 float32 防止溢出
                weight_A = state_dict[key_A].to(self.device).float()
                weight_B = state_dict[key_B].to(self.device).float()
                
                # 计算 ΔW = B @ A 
                # Peft: B is (d_out, r), A is (r, d_in) -> Result (d_out, d_in)
                delta_W = weight_B @ weight_A
                
                # 使用层名作为 key (去除 lora 后缀)
                layer_name = key.split(".lora_A")[0]
                delta_w_dict[layer_name] = delta_W
                
        return delta_w_dict

    def calculate_spectral_score(self, client_state_dict, global_state_dict, top_k=1):
        """
        [V2.0] 综合考量【方向一致性】和【能量合理性】
        """
        client_text_W = self.merge_lora_to_matrix(client_state_dict, "text_model")
        global_text_W = self.merge_lora_to_matrix(global_state_dict, "text_model")
        
        similarities = []
        
        for layer_name in client_text_W.keys():
            if layer_name not in global_text_W:
                continue
                
            W_c = client_text_W[layer_name]
            W_g = global_text_W[layer_name]
            
            try:
                # SVD 分解
                U_c, S_c, V_c = torch.svd_lowrank(W_c, q=top_k)
                U_g, S_g, V_g = torch.svd_lowrank(W_g, q=top_k)
                
                # --- 1. 方向得分 (Direction Score) ---范围 [0, 1]
                sim_u = torch.abs(F.cosine_similarity(U_c[:, 0], U_g[:, 0], dim=0)) 
                sim_v = torch.abs(F.cosine_similarity(V_c[:, 0], V_g[:, 0], dim=0))
                dir_score = (sim_u + sim_v) / 2.0
                
                ## 先不看能量!!!
                # --- 2. 能量惩罚 (Energy Penalty) ---
                # 逻辑：Label Shuffle 通常会导致参数更新幅度（奇异值）异常变大
                # 我们比较 Top-1 奇异值的大小
                energy_c = S_c[0]
                energy_g = S_g[0]
                
                # 计算比率：我们希望 energy_c 和 energy_g 数量级接近
                # 如果 energy_c 是 energy_g 的 5 倍，说明更新太激进，可能是攻击
                # 构造一个简单的惩罚函数: exp(-|log(ratio)|) -> 越接近 1 分数越高
                ratio = energy_c / (energy_g + 1e-6)
                
                # 设定一个容忍度，允许适当的个性化差异，但拒绝剧烈波动
                # 如果比率在 0.5 ~ 2.0 之间，penalty 接近 1；如果比率 > 5，penalty 接近 0
                if ratio > 3.0 or ratio < 0.33: 
                    energy_penalty = 0.1 # 重罚
                else:
                    energy_penalty = 1.0 # 放行
                
                # 也可以使用更平滑的惩罚： energy_score = 1 / (1 + abs(ratio - 1))
                
                # --- 3. 综合得分 ---
                # 如果能量异常，方向再对也没用
                layer_score = dir_score * energy_penalty
                similarities.append(layer_score.item())
                
            except Exception as e:
                print(f"[Warning] SVD Error at {layer_name}: {e}")
                continue
                
        if len(similarities) == 0:
            return 0.0
        return sum(similarities) / len(similarities)

    def aggregate(self, client_state_dicts: List[Dict], strategy='ours'):
        """
        聚合入口函数
        """
        if strategy == 'fedavg':
            print("Aggregating with Strategy: FedAvg")
            return self._fedavg(client_state_dicts)
        elif strategy == 'ours':
            print("Aggregating with Strategy: MA-COM (Spectral Merging)")
            return self._spectral_weighted_merge(client_state_dicts)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _fedavg(self, client_state_dicts):
        """标准的 FedAvg: 权重平均"""
        n_clients = len(client_state_dicts)
        avg_state_dict = copy.deepcopy(client_state_dicts[0])
        
        for key in avg_state_dict.keys():
            # 必须是 Tensor 才能平均
            if isinstance(avg_state_dict[key], torch.Tensor):
                # Start with the first client's params
                sum_param = client_state_dicts[0][key].clone().to(self.device)
                for i in range(1, n_clients):
                    sum_param += client_state_dicts[i][key].to(self.device)
                avg_state_dict[key] = sum_param / n_clients
                
        return avg_state_dict

    # def _spectral_weighted_merge(self, client_state_dicts):
    #     """
    #     1. 用 FedAvg 计算一个临时的 "Global Consensus"。
    #     2. 基于 Text LoRA 的 SVD 相似度计算每个 Client 的权重。
    #     3. 用算出来的权重聚合 Vision LoRA。
    #     4. Text LoRA 依然用 FedAvg (作为锚点)。
    #     """
    #     n_clients = len(client_state_dicts)
        
    #     # 1. 计算 Pseudo-Global (平均值) 作为参考系
    #     # 这里为了省内存，只计算 Text 部分的平均即可，但为了代码简单，先算全量平均
    #     global_anchor = self._fedavg(client_state_dicts)
        
    #     # 2. 计算每个 Client 的谱分数
    #     raw_scores = []
    #     for i, client_dict in enumerate(client_state_dicts):
    #         score = self.calculate_spectral_score(client_dict, global_anchor)
    #         raw_scores.append(score)
    #         print(f"  > Client {i} Spectral Score (Text Stability): {score:.4f}")
            
    #     # 3. 计算聚合权重 (Softmax)
    #     scores_tensor = torch.tensor(raw_scores).to(self.device)
    #     # 减去最大值防止溢出，除以温度系数拉大差距
    #     weights = F.softmax(scores_tensor / self.temperature, dim=0)
        
    #     print(f"  > Calculated Aggregation Weights: {weights.tolist()}")
        
    #     # 4. 加权聚合
    #     final_state_dict = copy.deepcopy(client_state_dicts[0])
        
    #     for key in final_state_dict.keys():
    #         if not isinstance(final_state_dict[key], torch.Tensor):
    #             continue
                
    #         # 初始化为 0
    #         weighted_param = torch.zeros_like(final_state_dict[key]).to(self.device)
            
    #         for i, client_dict in enumerate(client_state_dicts):
    #             param = client_dict[key].to(self.device)
                
    #             if "vision_model" in key:
    #                 # [关键创新] 视觉参数：使用“谱权重”进行纠正
    #                 # 如果 Text 烂了 (权重低)，Vision 参数也会被忽略
    #                 weighted_param += weights[i] * param
    #             else:
    #                 # 文本参数：为了稳妥，保持 FedAvg (或者也可以加权，但 FedAvg 更符合“锚点”定义)
    #                 weighted_param += (1.0 / n_clients) * param
            
    #         final_state_dict[key] = weighted_param
            
    #     return final_state_dict
    def _spectral_weighted_merge(self, client_state_dicts):
        """
        【修正后的算法】全量加权聚合 (All-in Spectral Weighted Merging)
        逻辑：
        1. 计算 Text LoRA 的谱相似度分数。
        2. 如果分数低，说明该 Client 语义崩塌（如 Label Shuffle）。
        3. 既然语义崩塌，那么该 Client 的【所有参数】（Vision 和 Text）都不可信。
        4. 使用计算出的权重聚合所有参数，彻底隔离恶意/低质客户端。
        """
        n_clients = len(client_state_dicts)
        
        # 1. 计算 Pseudo-Global (平均值) 作为参考系
        # 用于给每个 Client 找一个对比标杆
        global_anchor = self._fedavg(client_state_dicts)
        
        # 2. 计算每个 Client 的谱分数 (基于 Text LoRA SVD)
        raw_scores = []
        for i, client_dict in enumerate(client_state_dicts):
            score = self.calculate_spectral_score(client_dict, global_anchor)
            raw_scores.append(score)
            # 打印分数，用于观察谁是坏人
            print(f"  > Client {i} Spectral Score: {score:.4f}")
            
        # 3. 计算聚合权重 (Softmax)
        
        scores_tensor = torch.tensor(raw_scores).to(self.device)
        weights = F.softmax(scores_tensor / self.temperature, dim=0)
        
        print(f"  > Calculated Weights: {weights.tolist()}")
        
        # 4. 加权聚合 (关键修改：不再区分 vision/text，一视同仁)
        final_state_dict = copy.deepcopy(client_state_dicts[0])
        
        for key in final_state_dict.keys():
            # 只聚合 Tensor 类型的参数
            if not isinstance(final_state_dict[key], torch.Tensor):
                continue
                
            # 初始化为 0
            weighted_param = torch.zeros_like(final_state_dict[key]).to(self.device)
            
            for i, client_dict in enumerate(client_state_dicts):
                param = client_dict[key].to(self.device)
                
                # --- 核心修改点 ---
                # 删除 if "vision" 判断。
                # 无论这个参数属于 Vision 还是 Text，都乘上它的信誉权重？
                # 坏人 (weights[i] ≈ 0) 的所有参数都会被剔除。
                weighted_param += weights[i] * param
            
            final_state_dict[key] = weighted_param
            
        return final_state_dict