<!-- 本文件说明如何搭建并运行视觉发散联邦实验 -->
<!-- 阅读后可快速完成环境、数据与训练配置 -->

# myFedCLIP：视觉发散、本地 LoRA 训练基线

## 1. 环境准备

- 建议使用 Conda 创建独立环境：
  ```bash
  conda create -n myfedclip python=3.10 -y
  conda activate myfedclip
  pip install -r requirements.txt
  ```
- 依赖包含 PyTorch 2.1 + CUDA、Transformers、PEFT、Accelerate、Matplotlib 等。

## 2. 数据下载

1. Office-Home（约 3 GB）
   - 下载链接：https://www.hemanthdv.org/officeHomeDataset.html
   - 解压到 `/data1/lc/data/office_home/`，保持四个域子目录：`Art`、`Clipart`、`Product`、`Real_World`。
2. 可选 Unseen 域：DomainNet 子集
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

## 3. GPU 资源分配（3 × RTX 3090）

- GPU0：Client1(Art) + Client2(Clipart)
- GPU1：Client3(Product) + Client4(Real World)
- GPU2：Server（参数聚合、SVD 分析、未见域评估）

## 4. 运行流程

在运行脚本前请确保 Python 能找到 `src/` 路径，例如：

- Linux/macOS: `export PYTHONPATH=src`
- PowerShell: `$env:PYTHONPATH="src"`

1. **本地训练（无联邦聚合）**
   ```bash
   python -m train_clients --config configs/local.yaml
   ```
   - 每个客户端单独训练 5 epoch，间隔 100 step 记录 LoRA ΔW。
   - Checkpoint、LoRA 轨迹保存在 `checkpoints/{domain}/`（domain 取 Art/Clipart/Product/Real_World，对应小写下划线）。
   - 如需通过命令行指定域，请使用 `--domains Art Clipart Product Real_World`（与实际文件夹一致）。
2. **相似度分析 / Figure 1**
   ```bash
   python -m analyze_similarity \
     --client_a checkpoints/art \
     --client_b checkpoints/clipart \
     --output_fig outputs/figure1.png
   ```
   - 脚本会分别比较文本/视觉 LoRA，生成 CSV + 图像。
3. **谱共识聚合（服务器）**
   ```bash
   python -m server \
     --snapshots checkpoints/art/lora_art_step001000.pt checkpoints/clipart/lora_clipart_step001000.pt \
     --strategy spectral --top_k 4 --temperature 5.0 \
     --output checkpoints/global_spectral.pt
   ```
   - `--strategy fedavg` 可切换普通平均；`--snapshots` 接受任意数量客户端快照；输出文件只包含聚合后的 LoRA state，可由客户端重新加载。
4. **联邦消融主控**
   ```bash
   python -m main_fed \
     --config configs/local.yaml \
     --rounds 10 --local_steps 200 \
     --malicious_domains real_world \
     --strategy spectral
   ```
   - `main_fed.py` 读取配置，循环执行本地训练 + 聚合；通过 `--malicious_domains` 指定恶意客户端，测试 Spectral Consensus 的鲁棒性。

## 5. Spectral Consensus 聚合机制

1. **Global Reference**：对所有文本 LoRA 模块做 FedAvg，得到参考 ΔW。
2. **Spectral Score**：客户端文本 ΔW 与参考 ΔW 逐层做 SVD，计算前 `k` 个主成分子空间重合度，得到分数。
3. **Vision Reweighting**：将分数经过 Softmax(temperature) 归一化，只在视觉模态上按权重聚合；文本保持平均，充当语义锚点。
4. **回退策略**：若缺少文本模块或分数全零，自动退化为 FedAvg。

实现入口：`server.py` 中的 `get_delta_w`、`compute_subspace_similarity`、`execute_spectral_aggregation`。

## 6. 恶意语义攻击

- `ClipLoRALearner` 支持 `is_malicious` / `poison_shuffle_prob`，当启用时，会在 `train_step` 对 labels 做随机洗牌，实现“看图说反话”的语义攻击。
- 攻击只影响文本分支训练轨迹，视觉编码器仍围绕真实图片更新，有助于复现“文本稳定 vs 视觉发散”并验证谱共识的纠偏能力。
- 在 `main_fed.py` 中通过 `--malicious_domains` 指定攻击者（默认 `real_world`），也可在自定义脚本中直接设置。

## 7. 推荐实验步骤

1. **环境与镜像**：按第 1 节创建 Conda 环境；若处于国内网络，可先执行 `export HF_ENDPOINT=https://hf-mirror.com`（PowerShell: `$env:HF_ENDPOINT="https://hf-mirror.com"`）以加速下载 CLIP。
2. **阶段一（本地训练）**：运行 `python -m train_clients`，收集每个域的 LoRA 轨迹；随后用 `python -m analyze_similarity` 绘制 Figure 1 证明视觉发散。
3. **阶段二（谱分析）**：选择相同 step 的快照，调用 `python -m server --strategy spectral ...` 获得聚合模型，并与 `--strategy fedavg` 做对比，观察奇异值与子空间重合度。
4. **阶段三（联邦消融）**：执行 `python -m main_fed --rounds 10 --malicious_domains real_world --strategy spectral`，记录每轮日志与聚合权重，验证 Spectral Consensus 对恶意客户端的抑制效果。
5. **阶段四（评估与扩展）**：把 `global_spectral.pt` 或 `global_rounds/round_*.pt` 回载到客户端，评估 DomainNet 等未见域；可尝试调整 `top_k`、`temperature`、恶意比例等参数。

## 8. 后续扩展

- 在 `server.py` 中继续加入其它鲁棒聚合（如截断均值、RFA）或自适应温度策略。
- `main_fed.py` 可扩展成异步/分布式模拟器，或加入客户端采样、通信压缩等模块。
- DomainNet 子集可作为 GPU2 的未见域评估集，进一步验证谱共识对泛化性能的影响。

- `server.py` 将承载 ServerCoordinator：可插入 SVD、聚合策略、未见域测试。
- 可将 DomainNet 子集加载到 GPU2 上，用于评估各客户端参数在新域的鲁棒性。
- 完成 Figure 1 后，即可继续实现实际联邦通信与多轮聚合实验。
