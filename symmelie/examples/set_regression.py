"""Permutation-equivariant features and invariant pooling inspired by Deep Sets.

Run python -m examples.set_regression; sources and differences: examples/README.md.
"""
import torch
from torch import nn
from symmelie import EquivariantLinear, symmetric
from ._training import cli, fit


class SetRegressor(nn.Module):
    """Fixed-size scalar sets. Coordinatewise activations commute with permutations."""
    def __init__(self, nodes=4, channels=4):
        super().__init__()
        self.nodes, self.channels = nodes, channels
        _, slots = symmetric(nodes)
        hidden = slots.tensor_product(slots.trivial(channels))
        self.first = EquivariantLinear(slots, hidden)
        self.second = EquivariantLinear(hidden, hidden)
        self.readout = nn.Sequential(nn.Linear(channels, 16), nn.SiLU(), nn.Linear(16, 1)).double()

    def forward(self, values):
        if values.shape[-1] != self.nodes:
            raise ValueError(f"Expected {self.nodes} scalar set elements")
        features = torch.nn.functional.silu(self.first(values))
        features = torch.nn.functional.silu(self.second(features))
        pooled = features.unflatten(-1, (self.nodes, self.channels)).mean(-2)
        return self.readout(pooled).squeeze(-1)


def set_target(values):
    return values.square().mean(-1) + .3 * values.mean(-1)


def run(steps=250, seed=0):
    """Learn a set statistic using S4-constrained layers and invariant mean pooling."""
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed + 300)
    def sample(batch):
        values = torch.rand(batch, 4, generator=rng, dtype=torch.float64) * 2 - 1
        return values, set_target(values)
    model = SetRegressor()
    metrics = fit(model, sample, steps)
    values, _ = sample(8)
    with torch.no_grad():
        metrics["permutation_max_error"] = (model(values[..., [2, 0, 3, 1]]) - model(values)).abs().max().item()
    return metrics


if __name__ == "__main__":
    cli(run, steps=250)
