"""Configuration and reproducibility helpers for the WEF experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import json
import random

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEEDS = (7, 42, 73, 101, 314, 509, 1234, 2026, 2718, 4096)


@dataclass
class RewardConfig:
    """Dimensionless WEF reward weights.

    Each physical deficit is normalized by the corresponding monthly demand or
    capacity before these weights are applied.  This keeps GWh and 1e8 m3 from
    being compared numerically as though they shared a unit.
    """

    water: float = 1.0
    energy: float = 0.35
    food: float = 1.0
    import_share: float = 0.25
    # One end-of-year charge for depleting carry-over storage below the
    # episode's initial storage.  The normalized deficit is in [0, 1], making
    # a unit weight comparable to the other dimensionless sector penalties.
    terminal_storage: float = 1.0


@dataclass
class PhysicalConfig:
    """Hard physical safeguards applied before reward calculation.

    A finite one-year episode otherwise gives a policy an artificial incentive
    to empty reservoirs in December because the next year's loss is invisible.
    Requiring year-end storage to recover to the year-start level removes that
    boundary artefact.  A fraction of 1.0 is a no-net-depletion condition;
    setting it to 0.0 reduces the target to the physical minimum.
    """

    terminal_storage_target_fraction: float = 1.0


@dataclass
class PPOConfig:
    updates: int = 160
    rollout_episodes: int = 4
    update_epochs: int = 6
    minibatch_size: int = 48
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden_dim: int = 64
    gnn_layers: int = 2


@dataclass
class ValidationConfig:
    """Validation-only model selection shared by every learned method.

    The same zero-, medium-, and maximum-uniform-delay suite is used for both
    methods so that early stopping cannot favour one architecture through a
    method-specific selection rule. Test years are never consulted during
    training.
    """

    interval_updates: int = 10
    minimum_updates: int = 60
    patience_checks: int = 5
    minimum_improvement: float = 1.0e-4
    delay_scenarios: tuple[str, ...] = ("tau0", "tau3", "tau5")


@dataclass
class ExperimentConfig:
    data_path: Path = PROJECT_ROOT / "py" / "filled.csv"
    result_dir: Path = PROJECT_ROOT / "result"
    train_years: tuple[int, ...] = tuple(range(2001, 2016))
    validation_years: tuple[int, ...] = (2016, 2017, 2018)
    test_years: tuple[int, ...] = (2019, 2020, 2021)
    # Ten fixed, pre-declared seeds quantify optimization variability.  Their
    # values are arbitrary identifiers; they do not represent new river years.
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    # The final slot is retained as a zero-padded channel so the retained
    # checkpoints trained with no delayed observations remain compatible with the 24 x 12
    # observation contract.
    feature_dim: int = 12
    delay_levels: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    device: str = "cpu"
    reward: RewardConfig = field(default_factory=RewardConfig)
    physical: PhysicalConfig = field(default_factory=PhysicalConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return list(value)
            if hasattr(value, "__dataclass_fields__"):
                return {k: convert(v) for k, v in asdict(value).items()}
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            if isinstance(value, list):
                return [convert(v) for v in value]
            return value

        return convert(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def seed_everything(seed: int) -> None:
    """Make one experiment repeatable on the same PyTorch/platform build."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    torch.set_num_threads(1)
