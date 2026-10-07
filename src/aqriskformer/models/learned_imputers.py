"""Independent graph-imputation implementations for outage comparisons.

The modules in this file were written for this project from the method
descriptions in the GRIN and IGNNK papers.  They do not copy either authors'
reference implementation.  ``BidirectionalGraphImputer`` follows the principal
GRIN design choices (directional graph-recurrent reconstruction followed by a
bidirectional merge), whereas ``InductiveKrigingImputer`` is deliberately named
an IGNNK-*style* comparator because it uses the paper's three-layer residual
diffusion-GCN idea but is trained jointly with the downstream forecasting loss.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _random_walk(graph: torch.Tensor) -> torch.Tensor:
    """Return a finite row-normalized support without adding self loops."""
    graph = graph.float().clamp_min(0)
    return graph / graph.sum(-1, keepdim=True).clamp_min(1e-8)


class DiffusionGraphLinear(nn.Module):
    """Linear projection over forward/backward diffusion-polynomial features."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        graph: torch.Tensor,
        order: int = 2,
        activation: str | None = None,
    ) -> None:
        super().__init__()
        if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
            raise ValueError("graph must be a square matrix")
        if order < 1:
            raise ValueError("diffusion order must be positive")
        support = _random_walk(graph)
        self.register_buffer("forward_support", support)
        self.register_buffer("backward_support", _random_walk(graph.T))
        self.order = int(order)
        self.activation = activation
        self.projection = nn.Linear(input_size * (1 + 2 * self.order), output_size)

    @staticmethod
    def _propagate(support: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ij,...jf->...if", support, values)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        terms = [values]
        for support in (self.forward_support, self.backward_support):
            previous = values
            current = self._propagate(support, values)
            terms.append(current)
            for _ in range(2, self.order + 1):
                following = 2.0 * self._propagate(support, current) - previous
                terms.append(following)
                previous, current = current, following
        output = self.projection(torch.cat(terms, dim=-1))
        if self.activation == "relu":
            return F.relu(output)
        if self.activation == "gelu":
            return F.gelu(output)
        return output


class DiffusionGraphGRUCell(nn.Module):
    """GRU cell whose affine maps are diffusion graph convolutions."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        graph: torch.Tensor,
        order: int,
    ) -> None:
        super().__init__()
        self.gates = DiffusionGraphLinear(
            input_size + hidden_size, 2 * hidden_size, graph, order
        )
        self.candidate = DiffusionGraphLinear(
            input_size + hidden_size, hidden_size, graph, order
        )

    def forward(self, values: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        reset, update = self.gates(torch.cat((values, hidden), dim=-1)).chunk(2, dim=-1)
        reset, update = reset.sigmoid(), update.sigmoid()
        candidate = self.candidate(torch.cat((values, reset * hidden), dim=-1)).tanh()
        return update * hidden + (1.0 - update) * candidate


class _GraphRecurrentImputationDirection(nn.Module):
    def __init__(
        self,
        features: int,
        stations: int,
        hidden: int,
        graph: torch.Tensor,
        order: int,
    ) -> None:
        super().__init__()
        self.features = features
        self.hidden = hidden
        self.initial_state = nn.Parameter(torch.zeros(stations, hidden))
        self.temporal_estimate = nn.Linear(hidden, features)
        self.spatial_context = DiffusionGraphLinear(
            features * 2 + hidden, hidden, graph, order, activation="gelu"
        )
        self.spatial_estimate = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, features)
        )
        self.recurrent = DiffusionGraphGRUCell(features * 2, hidden, graph, order)

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, length, _, _ = values.shape
        hidden = self.initial_state[None].expand(batch_size, -1, -1)
        estimates: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        for step in range(length):
            observed_values = values[:, step]
            observed_mask = mask[:, step].to(values.dtype)
            temporal = self.temporal_estimate(hidden)
            first_fill = torch.where(mask[:, step], observed_values, temporal)
            context = self.spatial_context(
                torch.cat((first_fill, observed_mask, hidden), dim=-1)
            )
            spatial = self.spatial_estimate(torch.cat((hidden, context), dim=-1))
            completed = torch.where(mask[:, step], observed_values, spatial)
            hidden = self.recurrent(torch.cat((completed, observed_mask), dim=-1), hidden)
            estimates.append(spatial)
            states.append(hidden)
        return torch.stack(estimates, dim=1), torch.stack(states, dim=1)


class BidirectionalGraphImputer(nn.Module):
    """GRIN-method independent reimplementation for a fixed sensor graph."""

    def __init__(
        self,
        features: int,
        stations: int,
        hidden: int,
        graph: torch.Tensor,
        order: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.forward_direction = _GraphRecurrentImputationDirection(
            features, stations, hidden, graph, order
        )
        self.backward_direction = _GraphRecurrentImputationDirection(
            features, stations, hidden, graph, order
        )
        self.merge = nn.Sequential(
            nn.Linear(2 * features + 2 * hidden + features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, features),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        forward_estimate, forward_state = self.forward_direction(values, mask)
        backward_estimate, backward_state = self.backward_direction(
            values.flip(1), mask.flip(1)
        )
        backward_estimate = backward_estimate.flip(1)
        backward_state = backward_state.flip(1)
        merged = self.merge(
            torch.cat(
                (
                    forward_estimate,
                    backward_estimate,
                    forward_state,
                    backward_state,
                    mask.to(values.dtype),
                ),
                dim=-1,
            )
        )
        return torch.where(mask, values, merged)


class InductiveKrigingImputer(nn.Module):
    """IGNNK-style three-layer residual diffusion-GCN signal reconstructor."""

    def __init__(
        self,
        features: int,
        hidden: int,
        graph: torch.Tensor,
        order: int = 2,
    ) -> None:
        super().__init__()
        self.input_layer = DiffusionGraphLinear(
            2 * features, hidden, graph, order, activation="relu"
        )
        self.residual_layer = DiffusionGraphLinear(
            hidden, hidden, graph, order, activation="relu"
        )
        self.output_layer = DiffusionGraphLinear(hidden, features, graph, order)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat((values, mask.to(values.dtype)), dim=-1)
        first = self.input_layer(inputs)
        second = self.residual_layer(first) + first
        reconstruction = self.output_layer(second)
        return torch.where(mask, values, reconstruction)
