# src/main_fed.py
import argparse
import sys
from pathlib import Path

# 将项目根目录加入路径，防止 src 内部 import 报错
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.server import FedServer

def main():
    parser = argparse.ArgumentParser(description="Run Federated Learning Experiment")
    parser.add_argument("--config", type=str, default="configs/fed.yaml", help="Path to config file")
    args = parser.parse_args()

    # 初始化并运行联邦服务器
    # 所有的逻辑控制都在 Server 类里，这里只是一个启动按钮
    server = FedServer(args.config)
    server.run()

if __name__ == "__main__":
    main()