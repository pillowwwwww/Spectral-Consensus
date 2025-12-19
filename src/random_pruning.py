from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import torch

StateDict = Dict[str, torch.Tensor]


class RandomPruningAggregator:
    """Randomly prune LoRA parameters client-by-client, then FedAvg the survivors."""

    def __init__(
        self,
        prune_ratio: float = 0.5,
        seed: int | None = None,
        device: torch.device | str = "cuda",
    ) -> None:
        self.prune_ratio = min(max(prune_ratio, 0.0), 1.0)
        self.device = torch.device(device)

        self.rng = torch.Generator(device="cpu")
        if seed is None:
            # Non-deterministic seed for true randomness
            self.rng.seed()
        else:
            self.rng.manual_seed(seed)

    def _prune_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
        if self.prune_ratio == 0.0:
            return tensor.clone(), 0.0
        if self.prune_ratio == 1.0:
            return torch.zeros_like(tensor), 1.0

        keep_prob = 1.0 - self.prune_ratio
        mask = torch.rand(tensor.shape, generator=self.rng) < keep_prob
        mask = mask.to(tensor.device)

        pruned_tensor = tensor * mask
        pruned_fraction = 1.0 - mask.float().mean().item()
        return pruned_tensor, pruned_fraction

    def prune_state_dict(self, state_dict: StateDict) -> Tuple[StateDict, float]:
        pruned_state: StateDict = {}
        pruned_fracs: List[float] = []

        for name, param in state_dict.items():
            if not isinstance(param, torch.Tensor):
                pruned_state[name] = copy.deepcopy(param)
                continue

            if "lora_" not in name:
                pruned_state[name] = param.clone()
                continue

            pruned_param, pruned_fraction = self._prune_tensor(param)
            pruned_state[name] = pruned_param
            pruned_fracs.append(pruned_fraction)

        avg_pruned = sum(pruned_fracs) / len(pruned_fracs) if pruned_fracs else 0.0
        return pruned_state, avg_pruned

    def aggregate(self, client_state_dicts: List[StateDict]) -> StateDict:
        if not client_state_dicts:
            raise ValueError("client_state_dicts must not be empty for random pruning.")

        pruned_clients: List[StateDict] = []
        for idx, state in enumerate(client_state_dicts):
            pruned_state, pruned_fraction = self.prune_state_dict(state)
            pruned_clients.append(pruned_state)
            print(
                f"[RandomPruning] Client {idx}: pruned ~{pruned_fraction * 100:.2f}% "
                f"of LoRA params (target={self.prune_ratio * 100:.1f}%)"
            )

        return self._fedavg(pruned_clients)

    def _fedavg(self, state_dicts: List[StateDict]) -> StateDict:
        num_clients = len(state_dicts)
        avg_state = copy.deepcopy(state_dicts[0])

        for key, value in avg_state.items():
            if not isinstance(value, torch.Tensor):
                continue

            summed = state_dicts[0][key].to(self.device)
            for i in range(1, num_clients):
                summed = summed + state_dicts[i][key].to(self.device)
            avg_state[key] = summed / float(num_clients)

        return avg_state
