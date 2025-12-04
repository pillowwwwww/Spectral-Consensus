# 该目录下汇集 myFedCLIP 所有脚本（train_clients/main_fed 等）
# 设置 PYTHONPATH=src 后，可直接使用 `python -m train_clients` 方式运行

__all__ = [
    "data_loader",
    "learner",
    "train_clients",
    "analyze_similarity",
    "server",
    "main_fed",
    "utils",
]
