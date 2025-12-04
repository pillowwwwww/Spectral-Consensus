# src/scenario_b_heterogeneity.py
# -------------------------------------------------------------------------
# 场景 B：极端任务异构 (Extreme Task Heterogeneity)
# 目的：证明 SVD 能够识别并剔除“语义不相关”的任务，实现语义对齐。
# 设置：Client 0-2 为正常 Office-Home 任务；
#       Client 3 为模拟的 OOD 任务（强行修改标签名为医疗术语），制造语义空间正交。
# -------------------------------------------------------------------------
import argparse
import sys
import torch
import copy
from pathlib import Path

# 路径 hack，确保能导入 src 下的模块
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.learner import ClipLoRALearner, LearnerConfig
from src.spectral_merging import SpectralAggregator
from src.data_loader import create_officehome_dataloader

# === 模拟完全不相关的语义标签（医疗领域） ===
FAKE_MEDICAL_CLASSES = [
    "lung nodule", "fracture", "brain tumor", "pneumonia", "covid-19",
    "kidney stone", "liver cyst", "heart enlargement", "retinal damage", "skin lesion"
]

def run_heterogeneity_experiment():
    print(">>> 启动场景 B：极端异构性测试 (Medical Intruder) <<<")
    device = "cuda"
    
    # 1. 准备数据 (Office-Home)
    root_dir = "/data1/lc/data/office_home"
    domains = ["Art", "Clipart", "Product", "Real World"]
    
    # 2. 初始化客户端
    clients = []
    
    # Client 0-2: 正常客户端
    for i in range(3):
        domain = domains[i]
        loader, info = create_officehome_dataloader(root_dir, domain, batch_size=32)
        class_names = sorted(info.class_to_idx, key=info.class_to_idx.get)
        
        cfg = LearnerConfig()
        learner = ClipLoRALearner(class_names, device, {"r":16}, cfg, output_dir=f"checkpoints/scen_b/{domain}")
        clients.append({"learner": learner, "loader": loader, "name": domain, "type": "Normal"})

    # Client 3: OOD 客户端 (用 Real World 的图，但强制学医疗标签)
    # 这会迫使 Text Encoder 往完全不同的语义方向更新
    print(">>> 初始化 OOD 客户端：使用 Fake Medical Labels...")
    loader_ood, info_ood = create_officehome_dataloader(root_dir, "Real World", batch_size=32)
    
    # 截断或填充标签以匹配 loader 的输出 ID (这里简化处理，假设类别数够用，或者只取前几类)
    # 关键：Learner 里的 class_names 变成了医疗术语
    ood_classes = (FAKE_MEDICAL_CLASSES * 10)[:len(info_ood.class_to_idx)] 
    
    cfg = LearnerConfig()
    learner_ood = ClipLoRALearner(ood_classes, device, {"r":16}, cfg, output_dir="checkpoints/scen_b/Medical_OOD")
    clients.append({"learner": learner_ood, "loader": loader_ood, "name": "Medical_OOD", "type": "OOD"})

    # 3. 联邦循环
    aggregator = SpectralAggregator(device, temperature=0.05) # 低温以放大差异
    global_model = copy.deepcopy(clients[0]["learner"].extract_lora_delta("all"))
    
    # 只需要跑几轮就能看出现象
    for round_idx in range(5):
        print(f"\n--- Round {round_idx+1} ---")
        client_params = []
        
        for client in clients:
            # 下发参数
            client["learner"].load_lora_state(global_model)
            
            # 本地训练 (Train 1 epoch)
            # 注意：OOD 客户端虽然图是 RealWorld，但它的 Text LoRA 在学 "Lung Nodule"
            loss_sum = 0
            steps = 0
            for batch in client["loader"]:
                metrics = client["learner"].train_step(batch)
                loss_sum += metrics["loss"]
                steps += 1
                if steps >= 50: break # 加速实验，只跑 50 step
            
            print(f"Client [{client['name']}] ({client['type']}) Loss: {loss_sum/steps:.4f}")
            client_params.append(client["learner"].extract_lora_delta("all"))

        # 核心验证：谱聚合
        print(">>> 执行谱分析聚合...")
        # 注意观察输出的 Score
        global_model = aggregator.aggregate(client_params, strategy="ours")

if __name__ == "__main__":
    run_heterogeneity_experiment()