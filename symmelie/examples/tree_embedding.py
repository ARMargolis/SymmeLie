"""Hyperbolic hierarchy embedding inspired by Nickel & Kiela (2017).

Uses Lorentz coordinates and all-pairs stress, not the paper's ranking objective
or Riemannian optimizer. Run python -m examples.tree_embedding.
"""
import torch
from torch import nn
from symmelie import Hyperboloid, minkowski_dot, orthogonal
from ._training import cli


def tree_distances(depth=3):
    """Shortest-path distances in a complete binary tree (root indexed zero)."""
    if not isinstance(depth, int) or not 1 <= depth <= 6:
        raise ValueError("depth must be an integer between 1 and 6")
    nodes = 2 ** (depth + 1) - 1
    distances = torch.full((nodes, nodes), float('inf'), dtype=torch.float64)
    distances.fill_diagonal_(0)
    for child in range(1, nodes):
        parent = (child - 1) // 2
        distances[child, parent] = distances[parent, child] = 1
    for k in range(nodes):
        distances = torch.minimum(distances, distances[:, k:k+1] + distances[k:k+1, :])
    return distances


class TreeEmbedding(nn.Module):
    def __init__(self, nodes, dimension=2):
        super().__init__()
        self.spatial = nn.Parameter(torch.randn(nodes, dimension, dtype=torch.float64) * .1)
        self.geometry = Hyperboloid()

    def forward(self):
        # An unconstrained spatial chart guarantees valid upper-sheet points.
        # Adam optimizes this chart; the optimizer is not Lorentz-equivariant.
        return self.geometry.from_spatial(self.spatial)


def embedding_stress(model, target):
    points = model()
    # Select off-diagonal pairs before sqrt; distance is nondifferentiable at zero.
    indices = torch.triu_indices(points.shape[0], points.shape[0], offset=1, device=points.device)
    squared = model.geometry.squared_distance(points[indices[0]], points[indices[1]])
    distance = squared.clamp_min(1e-12).sqrt()
    return (distance - target[indices[0], indices[1]]).square().mean()


def run(steps=350, seed=0):
    """Fit 15 tree nodes in H2; report transductive reconstruction, not test accuracy."""
    torch.manual_seed(seed)
    target = tree_distances(3) * .5
    model = TreeEmbedding(target.shape[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=.05)
    initial = embedding_stress(model, target).item()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = embedding_stress(model, target)
        if not torch.isfinite(loss):
            raise RuntimeError("Embedding produced a nonfinite loss")
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        points = model()
        geometry = model.geometry
        _, lorentz = orthogonal(1, 2)
        transformed = lorentz.act(points.new_tensor([.2, -.3, .1]), points)
        before = geometry.squared_distance(points[:, None], points[None, :])
        after = geometry.squared_distance(transformed[:, None], transformed[None, :])
        return {"initial_reconstruction_stress": initial,
                "final_reconstruction_stress": embedding_stress(model, target).item(),
                "lorentz_distance_max_error": (before - after).abs().max().item(),
                "manifold_max_error": (minkowski_dot(points, points) + 1).abs().max().item(),
                "max_poincare_radius": geometry.to_poincare(points).norm(dim=-1).max().item()}


if __name__ == "__main__":
    cli(run, steps=350)
