# src/scenario_c_imbalance.py
# -------------------------------------------------------------------------
# 场景 C：数据不平衡/质量感知 (Data Imbalance / Quality Awareness)
# 目的：证明 SVD 能检测出“过拟合”或“低质量”的客户端。
# 设置：Client 0-2 数据充足；
#       Client 3 为 Few-Shot 客户端（只有 1 Batch 数据），但在本地疯狂过拟合。
# -------------------------------------------------------------------------
import argparse
import sys
import torch
import copy
from pathlib import Path
from torch.utils.data import Subset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.learner import ClipLoRALearner, LearnerConfig
from src.spectral_merging import SpectralAggregator
from src.data_loader import create_officehome_dataloader

def run_imbalance_experiment():
    print(">>> 启动场景 C：数据质量与过拟合测试 (Few-Shot Overfitting) <<<")
    device = "cuda"
    root_dir = "/data1/lc/data/office_home"
    domains = ["Art", "Clipart", "Product", "Real World"]
    
    clients = []
    
    # Client 0-2: 数据充足 (Full Data)
    for i in range(3):
        domain = domains[i]
        loader, info = create_officehome_dataloader(root_dir, domain, batch_size=32)
        class_names = sorted(info.class_to_idx, key=info.class_to_idx.get)
        
        learner = ClipLoRALearner(class_names, device, {"r":16}, LearnerConfig(), output_dir=f"checkpoints/scen_c/{domain}")
        clients.append({
            "learner": learner, 
            "loader": loader, 
            "name": domain, 
            "type": "Full-Data",
            "steps": 50 # 正常训练 50 步
        })

    # Client 3: Few-Shot (只有 32 张图) 但训练同样步数 -> 过拟合
    print(">>> 初始化 Few-Shot 客户端：仅使用 1 个 Batch 进行过拟合...")
    loader_full, info = create_officehome_dataloader(root_dir, "Real World", batch_size=32)
    
    # 强制只取前 32 张图
    dataset_few = Subset(loader_full.dataset, indices=range(32))
    loader_few = torch.utils.data.DataLoader(dataset_few, batch_size=32, shuffle=True)
    
    learner_few = ClipLoRALearner(class_names, device, {"r":16}, LearnerConfig(), output_dir="checkpoints/scen_c/FewShot")
    clients.append({
        "learner": learner_few, 
        "loader": loader_few, 
        "name": "Real_World_FewShot", 
        "type": "Few-Shot",
        "steps": 50 # 关键：数据少，但步数一样，强迫过拟合
    })

    # 3. 联邦循环
    # 注意：这里 temperature 可以稍微调高一点点，因为过拟合的区别比语义攻击要小
    aggregator = SpectralAggregator(device, temperature=0.1) 
    global_model = copy.deepcopy(clients[0]["learner"].extract_lora_delta("all"))
    
    for round_idx in range(5):
        print(f"\n--- Round {round_idx+1} ---")
        client_params = []
        
        for client in clients:
            client["learner"].load_lora_state(global_model)
            
            # 本地训练
            loss_sum = 0
            steps = 0
            
            # 无限循环 loader 直到满足 steps
            iterator = iter(client["loader"])
            for _ in range(client["steps"]):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(client["loader"])
                    batch = next(iterator)
                
                metrics = client["learner"].train_step(batch)
                loss_sum += metrics["loss"]
                steps += 1
            
            print(f"Client [{client['name']}] ({client['type']}) Loss: {loss_sum/steps:.4f}")
            client_params.append(client["learner"].extract_lora_delta("all"))

        # 核心验证
        print(">>> 执行谱分析聚合...")
        global_model = aggregator.aggregate(client_params, strategy="ours")

if __name__ == "__main__":
    run_imbalance_experiment()