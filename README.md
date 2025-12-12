<!-- 本文件说明如何搭建并运行视觉分散联邦实验 -->
<!-- 阅读后可快速完成环境、数据与训练配置 -->

# myFedCLIP：视觉分散、本地 LoRA 训练基线

## 1. 环境准备

- 建议使用 Conda 创建独立环境：
  ```bash
  conda create -n myfedclip python=3.10 -y
  conda activate myfedclip
  pip install -r requirements.txt
  ```
- 依赖包包括 PyTorch 2.1 + CUDA、Transformers、PEFT、Accelerate、Matplotlib 等。

## 2. 数据下载

1. Office-Home（约 3 GB）  
   - 下载链接：https://www.hemanthdv.org/officeHomeDataset.html  
   - 解压至 `/data1/lc/data/office_home/`，保持四个域子目录：`Art`、`Clipart`、`Product`、`Real_World`。
2. Unseen 域：DomainNet 子集  
   - 下载 `clipart` / `quickdraw` 等子域至 `/data1/lc/data/domainnet/`，供服务器侧评估使用。

目录示例：

```
myFedCLIP/
├── data/
│   ├── office_home/
│   │   ├── Art/
│   │   ├── Clipart/
│   │   ├── Product/
│   │   └── Real World/
│   └── domainnet/
├── configs/
│   └── local.yaml
├── checkpoints/
├── logs/
└── src/
    ├── __init__.py
    ├── train_clients.py
    ├── analyze_similarity.py
    ├── server.py
    ├── main_fed.py
    ├── learner.py
    ├── data_loader.py
    └── utils/
        └── logger.py
```

## 4. 运行流程

在运行脚本前请确保 Python 能找到 `src/` 路径，例如：

- Linux/macOS: `export PYTHONPATH=src`
- PowerShell: `$env:PYTHONPATH="src"`

1. **本地训练（无联邦聚合）**

   ```bash
   python -m train_clients --config configs/local.yaml
   ```

   - 每个客户端（Art / Clipart / Product / Real_World）独立训练若干步，周期性记录 LoRA ΔW。
   - Checkpoint 与 LoRA 轨迹保存在 `checkpoints/{domain}/`（`domain` 为小写+下划线形式）。
   - 如需通过命令行指定域，请使用  
     `--domains Art Clipart Product Real_World`（与实际文件夹一致）。

2. **LoRA 相似度分析**

   ```bash
   python -m analyze_similarity \
     --client_a checkpoints/art \
     --client_b checkpoints/clipart \
     --output_csv src/outputs/sim_art_clipart.csv \
     --output_fig src/outputs/sim_art_clipart.png
   ```

   用于绘制 “Text vs Vision LoRA 相似度” 曲线，观察视觉分散现象。

.............

## 推荐实验步骤

1. **环境与镜像**：按第 1 节创建 Conda 环境；若处于国内网络，可先执行  
   `export HF_ENDPOINT=https://hf-mirror.com`（PowerShell: `$env:HF_ENDPOINT="https://hf-mirror.com"`）以加速下 载 CLIP。
2. **阶段一（本地训练）**：运行  
   `python -m train_clients --config configs/local.yaml`，收集每个域的 LoRA 轨迹；随后用 `python -m analyze_similarity` 绘制 Figure 1，验证视觉分散。
3. **阶段二（联邦训练 / 谱分析）**：选择相同步数的 LoRA 快照，运行联邦脚本：
   - FedAvg AvgMerge 基线：`python -m src.main_fed --config src/configs/fed.yaml`
   - 谱聚合 A 版：`python -m src.main_fed --config src/configs/fed_ours.yaml`
   - 谱聚合 B 版：`python -m src.main_fed --config src/configs/fed_ours_b.yaml`

   对比 FedAvg 与各谱聚合策略在干净/含恶意客户端场景下的表现差异。

## 实验设计

本项目将所有“联邦 / 聚合算法”统一抽象为一个参数 `strategy`。  
通过 YAML 配置或命令行 `--strategy` 可以切换不同算法，方便在 FedAvg 基线与各种谱聚合策略之间切换。

### 1. 内置策略

所有策略都注册在 `src/strategies.py` 的 `STRATEGY_REGISTRY` 中：

- `fedavg`  
  对所有客户端上报的 LoRA ΔW `state_dict` 做逐参数算术平均：  
  这是“完全信任所有客户端”的 **AvgMerge 基线**。
- `ours` / `spectral_merging`  
  使用 `src/spectral_merging.py` 中的谱聚合算法（A 版）：  
  先用 FedAvg 得到 pseudo-global，再通过 SVD 计算每个客户端的光谱相似度，用 Softmax 得到权重，对所有 LoRA 参数做加权聚合。
- `ours_b` / `spectral_merging_b`  
  使用 `src/spectral_merging_b.py` 中的 B 版变体，保留相同接口但在评分 / 加权细节上有所调整，用于消融对比。

### 2. YAML 中配置策略

在联邦训练配置文件中，通过 `train.strategy` 选择算法，例如：

```yaml
train:
  strategy: "fedavg"          # 或 "ours" / "ours_b" / "spectral_merging" / "spectral_merging_b"
  rounds: 20
  epochs_per_round: 1
  model_name: "/data1/lc/models/clip-vit-b32"
  batch_size: 32
  learning_rate: 1.0e-4
  weight_decay: 0.01
  fp16: true
  device: "cuda"
  temperature: 0.1            # 谱聚合策略使用的温度（fedavg 不使用）
```
推荐直接在 YAML 中设置好 `train.strategy`，作为正式实验配置。

### 3. 命令行运行与策略覆盖

联邦训练入口为 `src/main_fed.py`：

```bash
# 依照 YAML 配置直接运行
python -m src.main_fed --config src/configs/fed.yaml
python -m src.main_fed --config src/configs/fed_ours.yaml
python -m src.main_fed --config src/configs/fed_ours_b.yaml

# 需要快速对比时，可以用 --strategy 覆盖 YAML 中的策略
python -m src.main_fed --config src/configs/fed_ours.yaml --strategy fedavg
python -m src.main_fed --config src/configs/fed.yaml --strategy spectral_merging_b
```
内部逻辑（`src/server.py`）会优先采用命令行传入的 `strategy`，若未提供则使用 YAML 中的 `train.strategy`：

```python
server = FedServer(args.config, strategy_override=args.strategy)
```

因此，**正常情况下只需要在 YAML 中设置策略**；`--strategy` 主要用于临时覆盖，方便做快速对照实验。

### 4. 客户端恶意标记与 AvgMerge 基线

联邦配置中的 `clients` 段用于指定哪些域是恶意客户端：

```yaml
clients:
  - domain: "Art"
    malicious: false
  - domain: "Clipart"
    malicious: false
  - domain: "Product"
    malicious: false
  - domain: "Real_World"
    malicious: true   # 本项目中，恶意端通过 label shuffle 产生“剧毒” LoRA
```
#### B 版：训练 / 测试阶段的 shuffle 与 malicious 对照

| 阶段 (Phase) | 客户端 (Domain)                           | Batch Shuffle (`shuffle`)       | Malicious Mode (`shuffle_labels`)      | 目的与逻辑 |
| ------------ | ------------------------------------------ | -------------------------------- | -------------------------------------- | ---------- |
| 训练 (Train) | Art, Clipart, Product（好人）             | True ✅（打乱顺序）             | False ❌（标签正常）                   | 正常学习：必须打乱数据顺序才能收敛；标签必须正确才能学到知识。 |
| 训练 (Train) | Real World（坏人 / 毒源）                 | True ✅（打乱顺序）             | True ✅（标签打乱）                    | 造毒过程：必须打乱数据顺序让它能“消化”；必须打乱标签让它强行记住错误的知识（过拟合噪音）。 |
| 测试 (Test)  | Art, Clipart, Product（好人）             | False ❌（顺序固定）            | False ❌（标签正常）                   | 常规体检：考试不打乱题目顺序（方便对号入座）；用标准答案（真标签）测真实能力。 |
| 测试 (Test)  | Real World（坏人 / 毒源，评估时视为干净） | False ❌（顺序固定）            | False ❌（标签正常）                   | 照妖镜时刻：测试时标签必须是正确的。由于它训练时学的是错的，现在用真标签考它，预期得分接近随机，从而证明它“中毒了”。 |

在 `src/server.py` 的 `_setup_clients` 中，这个标记会传入 `ClipLoRALearner` 构造函数，决定是否在 `train_step` 中进行标签打乱（`poison_shuffle_prob=1.0`）。

当配置为：

- 三个干净客户端 + 一个恶意客户端；
- `train.strategy: "fedavg"`；

得到的全局 LoRA 即为 **AvgMerge 基准线**：  
服务器对四个客户端一视同仁、简单平均，恶意客户端的剧毒更新会像墨汁一样严重污染全局模型。  
在此基础上，将 `strategy` 切换为 `"ours"` 或 `"ours_b"`，即可评估谱聚合是否能压制恶意客户端的影响、恢复性能。

### 5. 如何新增一个自定义策略

假设你希望添加一个名为 `"my_cool_algo"` 的新策略，大致步骤如下：

1. 在 `src/` 下新建文件 `my_cool_merging.py`，实现一个聚合函数：

   ```python
   # src/my_cool_merging.py
   from typing import Dict, List
   import torch

   StateDict = Dict[str, torch.Tensor]

   def my_cool_merge(client_state_dicts: List[StateDict]) -> StateDict:
       num = len(client_state_dicts)
       out = {k: v.clone() for k, v in client_state_dicts[0].items()}

       for key in out.keys():
           if not isinstance(out[key], torch.Tensor):
               continue
           acc = client_state_dicts[0][key]
           for i in range(1, num):
               acc = acc + client_state_dicts[i][key]
           out[key] = acc / float(num)
       return out
   ```

2. 在 `src/strategies.py` 中引入并注册该策略：

   ```python
   from src.my_cool_merging import my_cool_merge

   def my_cool_algo_strategy(client_state_dicts, device, cfg):
       del device, cfg
       return my_cool_merge(client_state_dicts)

   STRATEGY_REGISTRY["my_cool_algo"] = my_cool_algo_strategy
   ```

3. 在任一联邦 YAML 中使用：

   ```yaml
   train:
     strategy: "my_cool_algo"
     ...
   ```

之后即可通过：

```bash
python -m src.main_fed --config src/configs/fed_ours.yaml
# 或显式指定
python -m src.main_fed --config src/configs/fed_ours.yaml --strategy my_cool_algo
```

来运行你的自定义联邦算法。 

4. 评估
 python eval_global.py   --config src/configs/fed_ours_b.yaml   --checkpoint checkpoints/20251212_124454_checkpoints/global/global_round_1.pt

