"""Inertia regression inspired by EMLP's inertia task (Finzi et al., 2021).

Sources and differences: examples/README.md. Run python -m examples.inertia_tensor.
"""
import torch
from torch import nn
from symmelie import EquivariantBilinear, orthogonal_group
from ._training import cli, fit


def center(points, masses):
    weights = masses / masses.sum(-1, keepdim=True)
    return points - (weights.unsqueeze(-1) * points).sum(-2, keepdim=True)


def inertia_target(points, masses):
    """Moment of inertia about the center of mass, for positive point masses."""
    r = center(points, masses)
    eye = torch.eye(3, dtype=r.dtype, device=r.device)
    terms = r.square().sum(-1)[..., None, None] * eye - r.unsqueeze(-1) * r.unsqueeze(-2)
    return (masses[..., None, None] * terms).sum(-3)


class InertiaNet(nn.Module):
    def __init__(self):
        super().__init__()
        _, vector = orthogonal_group(3)
        self.coupling = EquivariantBilinear(vector, vector, vector.tensor_product(vector))
        # Learn the coefficients from scratch; no inertia formula in forward.
        nn.init.zeros_(self.coupling.linear.coefficients)

    def forward(self, points, masses):
        r = center(points, masses)
        contributions = self.coupling(r, r).unflatten(-1, (3, 3))
        return (masses[..., None, None] * contributions).sum(-3)


def run(steps=200, seed=0):
    """Learn an O(3)-equivariant rank-two tensor from unordered weighted points."""
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed + 100)
    def sample(batch):
        points = torch.randn(batch, 5, 3, generator=rng, dtype=torch.float64) * .5
        masses = torch.rand(batch, 5, generator=rng, dtype=points.dtype) + .2
        masses = masses / masses.sum(-1, keepdim=True)
        return points, masses, inertia_target(points, masses)
    model = InertiaNet()
    metrics = fit(model, sample, steps, lr=.05)
    points, masses, _ = sample(8)
    _, rep = orthogonal_group(3)
    rotation = rep.exp(torch.tensor([.2, -.3, .4], dtype=points.dtype)) @ rep.discrete_generators[0]
    shift = points.new_tensor([1., -2., .5])
    with torch.no_grad():
        output = model(points, masses)
        transformed = model(points @ rotation.T + shift, masses)
        metrics["equivariance_max_error"] = (transformed - rotation @ output @ rotation.T).abs().max().item()
    return metrics


if __name__ == "__main__":
    cli(run)
