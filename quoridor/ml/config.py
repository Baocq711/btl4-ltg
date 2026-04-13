"""Hyperparameter configuration for AlphaZero training."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class AlphaZeroConfig:
    # --- Network ---
    num_res_blocks: int = 4
    num_filters: int = 64
    value_hidden: int = 64
    input_channels: int = 8
    action_size: int = 209  # 81 move + 64 H-wall + 64 V-wall

    # --- MCTS ---
    num_simulations: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature_threshold: int = 15  # moves before switching to low temp

    # --- Training ---
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs_per_iter: int = 5
    replay_buffer_size: int = 50_000

    # --- Self-play ---
    num_parallel_games: int = 32
    num_games_per_iter: int = 100

    # --- Evaluation ---
    eval_games: int = 50
    win_rate_threshold: float = 0.55
    checkpoint_interval: int = 5

    # --- Paths ---
    checkpoint_dir: str = "checkpoints"
    tensorboard_dir: str = "runs"

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "AlphaZeroConfig":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid})
