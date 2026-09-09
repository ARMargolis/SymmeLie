"""Lorentz model H^n of curvature -1/radius^2, time coordinate first.

Inputs to geometric operations must be upper-sheet manifold points (or tangent
vectors where specified). Float64 is recommended, especially for large boosts.
"""
import math
import torch
from torch import nn


def minkowski_dot(x, y, keepdim=False):
    value = (x[..., 1:] * y[..., 1:]).sum(-1) - x[..., 0] * y[..., 0]
    return value.unsqueeze(-1) if keepdim else value


class Hyperboloid:
    def __init__(self, radius=1.0):
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("radius must be finite and positive")
        self.radius = float(radius)

    def from_spatial(self, spatial):
        """Coordinate chart, not an equivariant projection."""
        time = (self.radius ** 2 + spatial.square().sum(-1, keepdim=True)).sqrt()
        return torch.cat((time, spatial), dim=-1)

    def tangent_projection(self, x, v):
        return v + minkowski_dot(x, v, True) * x / self.radius ** 2

    def expmap(self, x, v):
        """Exponential at x; v must lie in the tangent space at x."""
        s = (minkowski_dot(v, v, True) / self.radius ** 2).clamp_min(0)
        small = s < 1e-6
        safe = s.clamp_min(1e-6).sqrt()
        cosh = torch.where(small, 1 + s / 2 + s.square() / 24, safe.cosh())
        sinhc = torch.where(small, 1 + s / 6 + s.square() / 120, safe.sinh() / safe)
        return cosh * x + sinhc * v

    def logmap(self, x, y):
        z = (-minkowski_dot(x, y, True) / self.radius ** 2 - 1).clamp_min(0)
        safe = z.clamp_min(1e-6)
        ratio = torch.where(z < 1e-6, 1 - z / 3 + 2 * z.square() / 15,
                            torch.acosh(1 + safe) / (safe * (safe + 2)).sqrt())
        return ratio * (y - (1 + z) * x)

    def squared_distance(self, x, y):
        """Smooth squared distance, with a series near coincident points."""
        z = (-minkowski_dot(x, y) / self.radius ** 2 - 1).clamp_min(0)
        safe = z.clamp_min(1e-6)
        square = torch.where(z < 1e-6, 2 * z - z.square() / 3 + 4 * z.pow(3) / 45,
                             torch.acosh(1 + safe).square())
        return self.radius ** 2 * square

    def distance(self, x, y):
        """Geodesic distance; nondifferentiable at coincidence. Train with squared_distance."""
        return self.squared_distance(x, y).sqrt()

    def to_poincare(self, x):
        return self.radius * x[..., 1:] / (x[..., :1] + self.radius)

    def from_poincare(self, u):
        s = u.square().sum(-1, keepdim=True) / self.radius ** 2
        if torch.any(s >= 1):
            raise ValueError("Points must lie strictly inside the Poincare ball")
        return torch.cat((self.radius * (1 + s) / (1 - s), 2 * u / (1 - s)), -1)


class HyperbolicAggregation(nn.Module):
    """Dense Lorentz-equivariant attention on (..., nodes, ambient_dim) points.

    Learn invariant distance weights, then normalize their positive weighted sum
    to the upper hyperboloid. This is an extrinsic centroid, not a Frechet mean.
    """
    def __init__(self, hidden_dim=16, radius=1.0):
        super().__init__()
        self.geometry = Hyperboloid(radius)
        self.score = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, points):
        d2 = self.geometry.squared_distance(points.unsqueeze(-2), points.unsqueeze(-3))
        weights = self.score(d2.unsqueeze(-1)).squeeze(-1).softmax(-1)
        mean = weights @ points
        norm2 = -minkowski_dot(mean, mean, True)
        return self.geometry.radius * mean / norm2.clamp_min(torch.finfo(mean.dtype).tiny).sqrt()
