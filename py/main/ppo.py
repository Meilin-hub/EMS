"""Compact, auditable PPO implementation for the monthly WEF environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn

from .config import PPOConfig, ValidationConfig
from .environment import YellowRiverWEFEnv
from .model import ActorCritic


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    episode_rewards: list[float]
    mean_water_cost: float
    mean_energy_cost: float
    mean_food_cost: float
    mean_terminal_storage_cost: float
    mean_storage_reserve_limited_release: float


@dataclass
class TrainingResult:
    history: list[dict[str, float]]
    best_update: int
    best_validation_score: float
    updates_completed: int
    stopped_early: bool


def _episode_advantages(
    rewards: list[float],
    values: list[float],
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    advantages = [0.0] * len(rewards)
    gae = 0.0
    next_value = 0.0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value - values[step]
        gae = delta + gamma * gae_lambda * gae
        advantages[step] = gae
        next_value = values[step]
    returns = [advantage + value for advantage, value in zip(advantages, values)]
    return advantages, returns


def collect_rollout(
    env: YellowRiverWEFEnv,
    model: ActorCritic,
    years: tuple[int, ...],
    episodes: int,
    delay_scenario: str,
    ppo: PPOConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> RolloutBatch:
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    log_probabilities: list[float] = []
    all_advantages: list[float] = []
    all_returns: list[float] = []
    episode_rewards: list[float] = []
    physical_records: list[dict[str, float]] = []

    model.eval()
    for _ in range(episodes):
        year = int(rng.choice(years))
        observation = env.reset(
            year,
            delay_scenario=delay_scenario,
            disturbance="normal",
        )
        episode_observations: list[np.ndarray] = []
        episode_actions: list[np.ndarray] = []
        episode_log_probabilities: list[float] = []
        rewards: list[float] = []
        values: list[float] = []
        done = False
        while not done:
            tensor_observation = torch.as_tensor(observation, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, log_probability, value = model.act(tensor_observation, deterministic=False)
            numpy_action = action.squeeze(0).cpu().numpy()
            next_observation, reward, done, info = env.step(numpy_action)
            episode_observations.append(observation.copy())
            episode_actions.append(numpy_action.copy())
            episode_log_probabilities.append(float(log_probability.item()))
            rewards.append(float(reward))
            values.append(float(value.item()))
            physical_records.append(info)
            observation = next_observation

        advantages, returns = _episode_advantages(rewards, values, ppo.gamma, ppo.gae_lambda)
        observations.extend(episode_observations)
        actions.extend(episode_actions)
        log_probabilities.extend(episode_log_probabilities)
        all_advantages.extend(advantages)
        all_returns.extend(returns)
        episode_rewards.append(float(sum(rewards)))

    return RolloutBatch(
        observations=torch.as_tensor(np.asarray(observations), dtype=torch.float32, device=device),
        actions=torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=device),
        old_log_probabilities=torch.as_tensor(log_probabilities, dtype=torch.float32, device=device),
        advantages=torch.as_tensor(all_advantages, dtype=torch.float32, device=device),
        returns=torch.as_tensor(all_returns, dtype=torch.float32, device=device),
        episode_rewards=episode_rewards,
        mean_water_cost=float(np.mean([r["cost_water_normalized"] for r in physical_records])),
        mean_energy_cost=float(np.mean([r["cost_energy_normalized"] for r in physical_records])),
        mean_food_cost=float(np.mean([r["cost_food_normalized"] for r in physical_records])),
        mean_terminal_storage_cost=float(
            np.mean(
                [
                    r["cost_terminal_storage_normalized"]
                    for r in physical_records
                    if int(r["month"]) == 12
                ]
            )
        ),
        mean_storage_reserve_limited_release=float(
            np.mean(
                [r["storage_reserve_limited_release_total"] for r in physical_records]
            )
        ),
    )


def validation_score(
    env: YellowRiverWEFEnv,
    model: ActorCritic,
    years: tuple[int, ...],
    delay_scenarios: tuple[str, ...],
    device: torch.device,
) -> float:
    """Mean deterministic monthly reward on validation years and scenarios."""

    rewards: list[float] = []
    model.eval()
    for delay_scenario in delay_scenarios:
        for year in years:
            observation = env.reset(
                year,
                delay_scenario=delay_scenario,
                disturbance="normal",
            )
            done = False
            while not done:
                tensor_observation = torch.as_tensor(
                    observation, dtype=torch.float32, device=device
                )
                with torch.no_grad():
                    action, _, _ = model.act(tensor_observation, deterministic=True)
                observation, reward, done, _ = env.step(action.squeeze(0).cpu().numpy())
                rewards.append(float(reward))
    return float(np.mean(rewards))


def validation_mean_episode_return(
    env: YellowRiverWEFEnv,
    model: ActorCritic,
    years: tuple[int, ...],
    delay_scenario: str,
    device: torch.device,
) -> float:
    """Mean deterministic annual return for one observation-delay scenario."""

    episode_returns: list[float] = []
    model.eval()
    for year in years:
        observation = env.reset(
            year,
            delay_scenario=delay_scenario,
            disturbance="normal",
        )
        rewards: list[float] = []
        done = False
        while not done:
            tensor_observation = torch.as_tensor(
                observation, dtype=torch.float32, device=device
            )
            with torch.no_grad():
                action, _, _ = model.act(tensor_observation, deterministic=True)
            observation, reward, done, _ = env.step(
                action.squeeze(0).cpu().numpy()
            )
            rewards.append(float(reward))
        episode_returns.append(float(sum(rewards)))
    return float(np.mean(episode_returns))


def train_policy(
    env: YellowRiverWEFEnv,
    model: ActorCritic,
    years: tuple[int, ...],
    ppo: PPOConfig,
    seed: int,
    validation_env: YellowRiverWEFEnv,
    validation_years: tuple[int, ...],
    validation: ValidationConfig,
    device: str = "cpu",
    progress: Callable[[int, int, dict[str, float]], None] | None = None,
    diagnostic_delay_scenarios: tuple[str, ...] = (),
    diagnostic_interval_updates: int = 1,
) -> TrainingResult:
    """Train one policy and restore the checkpoint selected on validation data."""

    torch_device = torch.device(device)
    model.to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo.learning_rate, eps=1.0e-5)
    if diagnostic_interval_updates < 1:
        raise ValueError("diagnostic_interval_updates must be at least one update")
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_update = 0
    best_score = float("-inf")
    checks_without_improvement = 0
    stopped_early = False

    for update in range(1, ppo.updates + 1):
        fraction = 1.0 - (update - 1.0) / max(ppo.updates, 1)
        optimizer.param_groups[0]["lr"] = ppo.learning_rate * fraction
        batch = collect_rollout(
            env, model, years, ppo.rollout_episodes, "tau0", ppo, torch_device, rng
        )
        advantages = batch.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1.0e-8)
        sample_count = len(batch.actions)
        losses: list[float] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        clip_fractions: list[float] = []

        model.train()
        for _ in range(ppo.update_epochs):
            permutation = torch.randperm(sample_count, device=torch_device)
            for start in range(0, sample_count, ppo.minibatch_size):
                index = permutation[start : start + ppo.minibatch_size]
                new_log_probability, entropy, value = model.evaluate_actions(
                    batch.observations[index], batch.actions[index]
                )
                log_ratio = new_log_probability - batch.old_log_probabilities[index]
                ratio = torch.exp(log_ratio)
                unclipped = ratio * advantages[index]
                clipped = torch.clamp(ratio, 1.0 - ppo.clip_ratio, 1.0 + ppo.clip_ratio) * advantages[index]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = 0.5 * (value - batch.returns[index]).pow(2).mean()
                entropy_mean = entropy.mean()
                loss = policy_loss + ppo.value_coef * value_loss - ppo.entropy_coef * entropy_mean

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), ppo.max_grad_norm)
                optimizer.step()

                losses.append(float(loss.detach().cpu()))
                policy_losses.append(float(policy_loss.detach().cpu()))
                value_losses.append(float(value_loss.detach().cpu()))
                entropies.append(float(entropy_mean.detach().cpu()))
                clip_fractions.append(float(((ratio - 1.0).abs() > ppo.clip_ratio).float().mean().detach().cpu()))

        diagnostics = {
            "update": update,
            "mean_episode_reward": float(np.mean(batch.episode_rewards)),
            "std_episode_reward": float(np.std(batch.episode_rewards)),
            "mean_water_cost": batch.mean_water_cost,
            "mean_energy_cost": batch.mean_energy_cost,
            "mean_food_cost": batch.mean_food_cost,
            "mean_terminal_storage_cost": batch.mean_terminal_storage_cost,
            "mean_storage_reserve_limited_release": (
                batch.mean_storage_reserve_limited_release
            ),
            "loss": float(np.mean(losses)),
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "clip_fraction": float(np.mean(clip_fractions)),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation_mean_monthly_reward": float("nan"),
            "best_validation_score": best_score,
            "validation_checks_without_improvement": float(checks_without_improvement),
        }

        should_validate = (
            update >= validation.minimum_updates
            and (update % validation.interval_updates == 0 or update == ppo.updates)
        )
        if should_validate:
            score = validation_score(
                validation_env,
                model,
                validation_years,
                validation.delay_scenarios,
                torch_device,
            )
            diagnostics["validation_mean_monthly_reward"] = score
            if score > best_score + validation.minimum_improvement:
                best_score = score
                best_update = update
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1
            diagnostics["best_validation_score"] = best_score
            diagnostics["validation_checks_without_improvement"] = float(
                checks_without_improvement
            )

        if diagnostic_delay_scenarios and update % diagnostic_interval_updates == 0:
            for delay_scenario in diagnostic_delay_scenarios:
                diagnostics[f"diagnostic_mean_episode_return_{delay_scenario}"] = (
                    validation_mean_episode_return(
                        validation_env,
                        model,
                        validation_years,
                        delay_scenario,
                        torch_device,
                    )
                )

        history.append(diagnostics)
        if progress is not None:
            progress(update, ppo.updates, diagnostics)
        if should_validate and checks_without_improvement >= validation.patience_checks:
            stopped_early = True
            break

    if best_state is None:
        best_update = len(history)
        best_score = validation_score(
            validation_env,
            model,
            validation_years,
            validation.delay_scenarios,
            torch_device,
        )
        best_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
    model.load_state_dict(best_state)
    model.to(torch_device)
    return TrainingResult(
        history=history,
        best_update=best_update,
        best_validation_score=best_score,
        updates_completed=len(history),
        stopped_early=stopped_early,
    )


def checkpoint_payload(
    model: ActorCritic,
    model_name: str,
    seed: int,
    feature_dim: int,
    hidden_dim: int,
    gnn_layers: int,
    variant_key: str = "",
    best_update: int = 0,
    best_validation_score: float = float("nan"),
    updates_completed: int = 0,
    stopped_early: bool = False,
) -> dict[str, object]:
    return {
        "model_name": model_name,
        "seed": seed,
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "gnn_layers": gnn_layers,
        "variant_key": variant_key,
        "best_update": best_update,
        "best_validation_score": best_validation_score,
        "updates_completed": updates_completed,
        "stopped_early": stopped_early,
        "state_dict": model.state_dict(),
    }
