"""Radial message passing inspired by EGNN (Satorras et al., 2021).

This is a single force-prediction block, not the full EGNN dynamics benchmark.
Run python -m examples.particle_forces; sources: examples/README.md.
"""
import torch
from torch import nn
from symmelie import EquivariantBilinear, orthogonal_group
from ._training import cli, fit


def pair_data(points, masses):
    relative = points.unsqueeze(-3) - points.unsqueeze(-2)  # x_j - x_i
    products = masses.unsqueeze(-1) * masses.unsqueeze(-2)
    return relative, products


def force_target(points, masses):
    relative, products = pair_data(points, masses)
    strength = products / (1 + relative.square().sum(-1)).pow(1.5)
    return (strength.unsqueeze(-1) * relative).mean(-2)


class ParticleForceNet(nn.Module):
    def __init__(self, hidden=24):
        super().__init__()
        _, vector = orthogonal_group(3)
        self.dot = EquivariantBilinear(vector, vector, vector.trivial())
        # Calibrate the solver's invariant basis to the ordinary dot product.
        basis = self.dot.linear.basis[0, 0].reshape(3, 3)
        with torch.no_grad():
            self.dot.linear.coefficients.fill_(3 / basis.trace())
        self.dot.requires_grad_(False)
        self.radial = nn.Sequential(nn.Linear(2, hidden), nn.SiLU(), nn.Linear(hidden, 1), nn.Softplus()).double()

    def forward(self, points, masses):
        relative, products = pair_data(points, masses)
        d2 = self.dot(relative, relative).squeeze(-1)
        strength = self.radial(torch.stack((d2, products), -1)).squeeze(-1)
        # Shared symmetric scalar weights times relative vectors give pairwise
        # antisymmetric forces. Self-messages vanish because their vectors are zero.
        return (strength.unsqueeze(-1) * relative).mean(-2)


def run(steps=250, seed=0):
    """Learn forces invariant to translation and equivariant to O(3) and node order."""
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed + 200)
    def sample(batch):
        points = torch.randn(batch, 5, 3, generator=rng, dtype=torch.float64) * .5
        masses = torch.rand(batch, 5, generator=rng, dtype=points.dtype) + .5
        return points, masses, force_target(points, masses)
    model = ParticleForceNet()
    metrics = fit(model, sample, steps)
    points, masses, _ = sample(8)
    _, rep = orthogonal_group(3)
    q = rep.exp(points.new_tensor([.4, -.2, .1])) @ rep.discrete_generators[0]
    with torch.no_grad():
        out = model(points, masses)
        changed = model(points @ q.T + points.new_tensor([2., -1., .7]), masses)
        metrics["equivariance_max_error"] = (changed - out @ q.T).abs().max().item()
        metrics["net_force_max_error"] = out.sum(-2).abs().max().item()
    return metrics


if __name__ == "__main__":
    cli(run, steps=250)
