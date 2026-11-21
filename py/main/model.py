"""Graph-attention and flat ablation actor-critic networks."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.distributions import Beta, Independent
from torch.nn import functional as F

from .graph import GRAPH, GraphSpec


class HeterogeneousGraphAttentionLayer(nn.Module):
    """Type-specific projection followed by relation-aware neighbour attention."""

    def __init__(self, input_dim: int, hidden_dim: int, graph: GraphSpec, dropout: float = 0.05) -> None:
        super().__init__()
        self.num_nodes = graph.num_nodes
        self.type_linears = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim) for _ in range(graph.num_types)]
        )
        self.relation_embedding = nn.Embedding(graph.num_relations, hidden_dim)
        self.node_embedding = nn.Embedding(graph.num_nodes, hidden_dim)
        self.attention = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("type_ids", graph.type_ids.clone())
        self.register_buffer("edge_sources", graph.edge_index[0].clone())
        self.register_buffer("edge_destinations", graph.edge_index[1].clone())
        self.register_buffer("edge_types", graph.edge_types.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3 or x.shape[1] != self.num_nodes:
            raise ValueError(f"Expected [batch,{self.num_nodes},features], received {tuple(x.shape)}")

        projected = torch.stack(
            [self.type_linears[int(self.type_ids[node])](x[:, node]) for node in range(self.num_nodes)],
            dim=1,
        )
        projected = F.gelu(projected + self.node_embedding.weight.unsqueeze(0))
        aggregates: list[torch.Tensor] = []
        for destination in range(self.num_nodes):
            edge_mask = self.edge_destinations == destination
            sources = self.edge_sources[edge_mask]
            relations = self.edge_types[edge_mask]
            messages = projected[:, sources] + self.relation_embedding(relations).unsqueeze(0)
            target = projected[:, destination].unsqueeze(1).expand_as(messages)
            scores = self.attention(torch.cat([target, messages, target * messages], dim=-1)).squeeze(-1)
            weights = torch.softmax(scores, dim=-1)
            aggregates.append(torch.sum(weights.unsqueeze(-1) * messages, dim=1))
        aggregated = torch.stack(aggregates, dim=1)
        return self.norm(projected + self.dropout(F.gelu(self.output(aggregated))))


class ActorCritic(nn.Module, ABC):
    action_dim = 4

    @abstractmethod
    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _distribution_and_value(self, observation: torch.Tensor) -> tuple[Independent, torch.Tensor]:
        latent = self.encode(observation)
        parameters = F.softplus(self.actor(latent)).view(-1, self.action_dim, 2) + 1.0
        distribution = Independent(Beta(parameters[..., 0], parameters[..., 1]), 1)
        value = self.critic(latent).squeeze(-1)
        return distribution, value

    def act(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self._distribution_and_value(observation)
        if deterministic:
            base = distribution.base_dist
            action = base.concentration1 / (base.concentration1 + base.concentration0)
        else:
            action = distribution.rsample()
        action = action.clamp(1.0e-5, 1.0 - 1.0e-5)
        return action, distribution.log_prob(action), value

    def evaluate_actions(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self._distribution_and_value(observation)
        safe_action = action.clamp(1.0e-5, 1.0 - 1.0e-5)
        return distribution.log_prob(safe_action), distribution.entropy(), value


class HeteroGNNActorCritic(ActorCritic):
    """Graph attention with topology-aligned action heads and a Beta policy.

    A single global pooled vector made the original GNN discard the identity of
    the four controlled assets just before producing their actions.  The revised
    policy combines global context with the embedding of W1, W2, W7, or W8 for
    the corresponding release/diversion action.  This is an architectural prior,
    not a method-specific reward change: every policy is still trained with the
    same PPO loss, observations, feasibility layer, and validation criterion.
    """

    def __init__(
        self,
        feature_dim: int = 12,
        hidden_dim: int = 64,
        layers: int = 2,
        graph: GraphSpec = GRAPH,
    ) -> None:
        super().__init__()
        graph_layers: list[nn.Module] = []
        for layer in range(layers):
            graph_layers.append(
                HeterogeneousGraphAttentionLayer(
                    feature_dim if layer == 0 else hidden_dim,
                    hidden_dim,
                    graph,
                )
            )
        self.graph_layers = nn.ModuleList(graph_layers)
        self.pool_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1, bias=False)
        )
        self.trunk = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.control_nodes = (0, 1, 6, 7)  # W1, W2, W7, W8
        self.action_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, 2),
                )
                for _ in self.control_nodes
            ]
        )
        self.critic = nn.Linear(hidden_dim, 1)
        self._initialize()

    def _initialize(self) -> None:
        for head in self.action_heads:
            output = head[-1]
            nn.init.orthogonal_(output.weight, gain=0.01)
            nn.init.zeros_(output.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def _encode_graph(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = observation.float()
        if x.ndim == 2:
            x = x.unsqueeze(0)
        for layer in self.graph_layers:
            x = layer(x)
        weights = torch.softmax(self.pool_score(x).squeeze(-1), dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * x, dim=1)
        return x, self.trunk(pooled)

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        _, pooled = self._encode_graph(observation)
        return pooled

    def _distribution_and_value(self, observation: torch.Tensor) -> tuple[Independent, torch.Tensor]:
        nodes, pooled = self._encode_graph(observation)
        parameters = torch.stack(
            [
                F.softplus(head(torch.cat([pooled, nodes[:, node]], dim=-1))) + 1.0
                for head, node in zip(self.action_heads, self.control_nodes)
            ],
            dim=1,
        )
        distribution = Independent(Beta(parameters[..., 0], parameters[..., 1]), 1)
        value = self.critic(pooled).squeeze(-1)
        return distribution, value


class PPOActorCritic(ActorCritic):
    """Conventional feed-forward PPO actor-critic baseline.

    This is deliberately named for the algorithm used in the manuscript rather
    than for an implementation detail of its encoder.  It receives the same
    24-node observation tensor as the GNN policies, but it does not perform
    graph message passing or use node-type/relation information.
    """

    def __init__(self, nodes: int = 24, feature_dim: int = 12, hidden_dim: int = 64) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nodes * feature_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.Tanh(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, self.action_dim * 2)
        self.critic = nn.Linear(hidden_dim, 1)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        x = observation.float()
        if x.ndim == 2:
            x = x.unsqueeze(0)
        return self.trunk(x)


def build_model(name: str, feature_dim: int, hidden_dim: int, gnn_layers: int) -> ActorCritic:
    if name == "gnn":
        return HeteroGNNActorCritic(feature_dim, hidden_dim, gnn_layers)
    if name == "ppo":
        return PPOActorCritic(GRAPH.num_nodes, feature_dim, hidden_dim)
    raise ValueError(f"Unknown model: {name}")
