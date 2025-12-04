import os
import torch
import copy
import yaml
import logging
from pathlib import Path
from src.learner import ClipLoRALearner, LearnerConfig
from src.spectral_merging import SpectralAggregator
from src.data_loader import create_officehome_dataloader

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FedServer")

class FedServer:
    def __init__(self, config_path: str):
        self.cfg = self._load_config(config_path)
        self.device = torch.device(self.cfg['train'].get('device', 'cuda'))
        
        # 初始化聚合器
        self.aggregator = SpectralAggregator(
            device=self.device, 
            temperature=self.cfg['train'].get('temperature', 0.1)
        )
        
        # 准备客户端
        self.clients = []
        self.client_loaders = []
        self._setup_clients()
        
        # 初始化全局模型 (取第一个客户端的初始参数作为全局初始)
        self.global_model_state = copy.deepcopy(self.clients[0].extract_lora_delta("all"))
        
        logger.info(f"Server initialized with {len(self.clients)} clients. Strategy: {self.cfg['train']['strategy']}")

    def _load_config(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _setup_clients(self):
        """初始化所有客户端实例和数据加载器"""
        data_cfg = self.cfg['data']
        train_cfg = self.cfg['train']
        lora_cfg = self.cfg['lora']
        client_list = self.cfg['clients'] # list of {domain: str, malicious: bool}

        for client_info in client_list:
            domain = client_info['domain']
            is_malicious = client_info.get('malicious', False)
            
            # 1. 准备数据
            dataloader, dataset_info = create_officehome_dataloader(
                root=data_cfg["office_home_root"],
                domain=domain,
                batch_size=train_cfg["batch_size"],
                num_workers=4,
                shuffle=True
            )   
            self.client_loaders.append(dataloader)
            
            # 2. 准备 Learner
            # 获取所有类名用于初始化 CLIP text embedding
            class_names = sorted(dataset_info.class_to_idx, key=dataset_info.class_to_idx.get)
            
            learner_cfg = LearnerConfig(
                model_name=train_cfg['model_name'],
                learning_rate=train_cfg['learning_rate'],
                weight_decay=train_cfg['weight_decay'],
                fp16=train_cfg['fp16']
            )
            
            learner = ClipLoRALearner(
                class_names=class_names,
                device=self.device,
                lora_cfg=lora_cfg,
                learner_cfg=learner_cfg,
                output_dir=f"checkpoints/fed/{domain}",
                is_malicious=is_malicious,  # 传入恶意标记
                poison_shuffle_prob=1.0 if is_malicious else 0.0 # 恶意客户端100%打乱标签
            )
            self.clients.append(learner)
            logger.info(f"Client [{domain}] initialized. Malicious: {is_malicious}")

    def run(self):
        rounds = self.cfg['train']['rounds']
        epochs_per_round = self.cfg['train']['epochs_per_round']
        strategy = self.cfg['train']['strategy']
        
        for r in range(rounds):
            logger.info(f"======== Round {r+1} / {rounds} ========")
            
            client_weights = []
            client_losses = []
            
            # --- 1. 客户端本地训练 ---
            for i, client in enumerate(self.clients):
                # A. 下发全局参数 (加载 LoRA 权重)
                client.load_lora_state(self.global_model_state)
                
                # B. 本地训练
                loader = self.client_loaders[i]
                loss_avg = 0
                steps = 0
                
                # 简单起见，每个 Round 训练一定的步数或 Epoch
                # 这里简单实现为训练 N 个 Epoch
                for _ in range(epochs_per_round):
                    for batch in loader:
                        metrics = client.train_step(batch)
                        loss_avg += metrics['loss']
                        steps += 1
                
                loss_avg /= max(steps, 1)
                client_losses.append(loss_avg)
                
                # C. 提取训练后的参数 (只提取 LoRA 部分)
                client_weights.append(client.extract_lora_delta("all"))
                
                logger.info(f"  Client {i} Train Loss: {loss_avg:.4f}")

            # --- 2. 服务器聚合 ---
            logger.info("Aggregating parameters...")
            
            # 调用 spectral_merging.py 中的核心逻辑
            self.global_model_state = self.aggregator.aggregate(
                client_weights, 
                strategy=strategy
            )
            
            # --- 3. (可选) 评估全局模型 ---
            # 可以在这里让某个 Client 加载最新的 Global Model 然后在测试集上跑一下
            # 这里简单打印一下完成日志
            if (r + 1) % 5 == 0:
                self._save_global_model(r+1)

    def _save_global_model(self, round_num):
        path = f"checkpoints/fed/global_round_{round_num}.pt"
        torch.save(self.global_model_state, path)
        logger.info(f"Saved global model to {path}")

# 入口函数在 main_fed.py 中调用