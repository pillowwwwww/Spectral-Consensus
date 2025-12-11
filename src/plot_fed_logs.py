import re
import matplotlib.pyplot as plt
import sys
import os

def parse_log_file(file_path):
    """
    解析日志文件，提取 Round 和 Client Loss 信息。
    """
    data = {}  # 格式: {client_id: {'rounds': [], 'losses': []}}
    current_round = 0
    
    # 正则表达式匹配
    round_pattern = re.compile(r"======== Round (\d+) / \d+ ========")
    loss_pattern = re.compile(r"Client (\d+) Train Loss: ([\d\.]+)")
    
    print(f"正在解析日志文件: {file_path} ...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 1. 检测 Round 信息
            round_match = round_pattern.search(line)
            if round_match:
                current_round = int(round_match.group(1))
                continue
            
            # 2. 检测 Client Loss 信息
            loss_match = loss_pattern.search(line)
            if loss_match:
                client_id = int(loss_match.group(1))
                loss_val = float(loss_match.group(2))
                
                if client_id not in data:
                    data[client_id] = {'rounds': [], 'losses': []}
                
                # 记录数据
                data[client_id]['rounds'].append(current_round)
                data[client_id]['losses'].append(loss_val)

    return data

def plot_client_losses(data):
    """
    绘制所有客户端的 Loss 曲线
    """
    plt.figure(figsize=(10, 6))
    
    # 定义颜色，确保 Client 3 (Real World) 显眼
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'] # 蓝，绿，橙，红
    markers = ['o', 's', '^', 'x']
    
    sorted_client_ids = sorted(data.keys())
    
    for i, cid in enumerate(sorted_client_ids):
        rounds = data[cid]['rounds']
        losses = data[cid]['losses']
        
        label = f"Client {cid}"
        if cid == 3:
            label += " (Real World / Noisy)"
            linewidth = 2.5
            linestyle = '--'
        else:
            linewidth = 1.5
            linestyle = '-'
            
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        plt.plot(rounds, losses, label=label, 
                 color=color, linewidth=linewidth, linestyle=linestyle, marker=marker, markersize=4)

    plt.title("FedAvg Training Loss per Client (Clean Flip Experiment)", fontsize=14)
    plt.xlabel("Communication Round", fontsize=12)
    plt.ylabel("Training Loss", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # 保存图片
    save_path = "fed_loss_curve.png"
    plt.savefig(save_path, dpi=300)
    print(f"图表已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    # 默认读取当前目录下的 training.log，也可以通过命令行参数传入
    log_file = "training.log" 
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
        
    if not os.path.exists(log_file):
        # 如果没有文件，创建一个临时的示例文件（基于你提供的数据）用于演示
        print(f"未找到 {log_file}，正在生成示例数据...")
        with open("demo_log.txt", "w") as f:
            f.write("""
2025-12-08 15:19:53,078 - FedServer - INFO - ======== Round 18 / 20 ========
2025-12-08 15:20:05,181 - FedServer - INFO -    Client 0 Train Loss: 0.0970
2025-12-08 15:20:26,148 - FedServer - INFO -    Client 1 Train Loss: 0.1962
2025-12-08 15:20:46,992 - FedServer - INFO -    Client 2 Train Loss: 0.0382
2025-12-08 15:21:24,964 - FedServer - INFO -    Client 3 Train Loss: 4.6530
2025-12-08 15:21:24,989 - FedServer - INFO - ======== Round 19 / 20 ========
2025-12-08 15:21:36,976 - FedServer - INFO -    Client 0 Train Loss: 0.0895
2025-12-08 15:21:57,871 - FedServer - INFO -    Client 1 Train Loss: 0.1812
2025-12-08 15:22:18,896 - FedServer - INFO -    Client 2 Train Loss: 0.0319
2025-12-08 15:22:59,257 - FedServer - INFO -    Client 3 Train Loss: 4.3008
2025-12-08 15:22:59,282 - FedServer - INFO - ======== Round 20 / 20 ========
2025-12-08 15:23:11,284 - FedServer - INFO -    Client 0 Train Loss: 0.0888
2025-12-08 15:23:32,096 - FedServer - INFO -    Client 1 Train Loss: 0.1725
2025-12-08 15:23:53,017 - FedServer - INFO -    Client 2 Train Loss: 0.0281
2025-12-08 15:24:30,232 - FedServer - INFO -    Client 3 Train Loss: 4.8348
            """)
        log_file = "demo_log.txt"

    client_data = parse_log_file(log_file)
    if client_data:
        plot_client_losses(client_data)
    else:
        print("未在日志中解析到有效数据，请检查日志格式。")