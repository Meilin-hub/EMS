"""The fixed 24-node heterogeneous graph defined by the manuscript."""

from __future__ import annotations

from dataclasses import dataclass

import torch


NODE_NAMES = (
    "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9",
    "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9",
    "F1", "F2", "F3", "F4", "F5", "F6",
)
NODE_TYPES = (
    "res", "res", "river", "river", "river", "river", "delivery", "delivery", "ctrl",
    "hp", "hp", "grid", "grid", "grid", "grid", "grid", "pump", "pump",
    "crop", "crop", "crop", "crop", "crop", "crop",
)
TYPE_NAMES = ("res", "river", "delivery", "ctrl", "hp", "grid", "pump", "crop")
RELATION_NAMES = ("water", "electricity", "irrigation", "information", "self")

# Experiments 3 and 4 use these three disjoint, exhaustive sets to define an
# observation delay.  A sector delay never changes the physical simulator: it
# only makes the selected nodes' observations stale before they enter the
# policy.  The names follow the WEF network rather than an economic sector
# classification: W7/W8 are water-delivery assets and F1--F6 represent
# irrigation-demand (food) nodes.
SECTOR_NODE_NAMES = {
    "water": NODE_NAMES[0:9],
    "energy": NODE_NAMES[9:18],
    "food": NODE_NAMES[18:24],
}


def node_indexes(node_names: tuple[str, ...] | list[str]) -> tuple[int, ...]:
    """Return graph indexes for named nodes, rejecting unknown identifiers."""

    lookup = {name: index for index, name in enumerate(NODE_NAMES)}
    unknown = sorted(set(node_names).difference(lookup))
    if unknown:
        raise ValueError(f"Unknown WEF node(s): {', '.join(unknown)}")
    return tuple(lookup[name] for name in node_names)


@dataclass(frozen=True)
class GraphSpec:
    node_names: tuple[str, ...]
    node_types: tuple[str, ...]
    type_ids: torch.Tensor
    edge_index: torch.Tensor
    edge_types: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return len(self.node_names)

    @property
    def num_types(self) -> int:
        return len(TYPE_NAMES)

    @property
    def num_relations(self) -> int:
        return len(RELATION_NAMES)


def build_graph() -> GraphSpec:
    """Create physical edges plus reverse information links and self-loops.

    Edge direction is source -> receiver.  Reverse information edges represent
    monitoring dependencies and do not create reverse physical flows.
    """

    idx = {name: i for i, name in enumerate(NODE_NAMES)}
    edges: list[tuple[int, int, int]] = []
    rel = {name: i for i, name in enumerate(RELATION_NAMES)}

    def add(source: str, target: str, relation: str, reverse_information: bool = True) -> None:
        edges.append((idx[source], idx[target], rel[relation]))
        if reverse_information:
            edges.append((idx[target], idx[source], rel["information"]))

    # Directed water chain in the paper.
    water_chain = ("W1", "W3", "W2", "W4", "W7", "W5", "W8", "W6", "W9")
    for source, target in zip(water_chain[:-1], water_chain[1:]):
        add(source, target, "water")

    # Release-to-power conversion and regional electricity clearing.
    add("W1", "E1", "electricity")
    add("W2", "E2", "electricity")
    for source in ("E1", "E2", "E3", "E4"):
        add(source, "E5", "electricity")
    add("E5", "E6", "electricity")
    add("E5", "E7", "electricity")
    add("E6", "E8", "electricity")
    add("E7", "E9", "electricity")

    # District water/pump interfaces and crop zones.
    add("W7", "E8", "irrigation")
    add("W8", "E9", "irrigation")
    for crop in ("F1", "F2", "F3"):
        add("W7", crop, "irrigation")
        add("E8", crop, "electricity")
    for crop in ("F4", "F5", "F6"):
        add("W8", crop, "irrigation")
        add("E9", crop, "electricity")

    for node in NODE_NAMES:
        edges.append((idx[node], idx[node], rel["self"]))

    edge_index = torch.tensor([(s, d) for s, d, _ in edges], dtype=torch.long).t().contiguous()
    edge_types = torch.tensor([r for _, _, r in edges], dtype=torch.long)
    type_ids = torch.tensor([TYPE_NAMES.index(t) for t in NODE_TYPES], dtype=torch.long)
    return GraphSpec(NODE_NAMES, NODE_TYPES, type_ids, edge_index, edge_types)


GRAPH = build_graph()
