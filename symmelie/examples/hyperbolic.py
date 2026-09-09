"""Run from the project root: python -m examples.hyperbolic."""
import torch
from symmelie import Hyperboloid, HyperbolicAggregation, orthogonal


def main():
    torch.manual_seed(0)
    geometry = Hyperboloid()
    points = geometry.from_spatial(torch.randn(8, 6, 3, dtype=torch.float64) * .3)
    model = HyperbolicAggregation().double()
    optimizer = torch.optim.Adam(model.parameters(), lr=.02)
    # Toy reconstruction objective: learn to retain each point during aggregation.
    initial = geometry.squared_distance(model(points), points).mean().item()
    for _ in range(60):
        optimizer.zero_grad()
        loss = geometry.squared_distance(model(points), points).mean()
        loss.backward()
        optimizer.step()
    _, rep = orthogonal(1, 3)
    boost = torch.tensor([.4, -.2, .1, .1, 0., -.1], dtype=points.dtype)
    error = (model(rep.act(boost, points)) - rep.act(boost, model(points))).abs().max()
    final = geometry.squared_distance(model(points), points).mean().item()
    print(f"Reconstruction loss: {initial:.6f} -> {final:.6f}")
    print(f"Lorentz equivariance max error: {error.item():.3e}")


if __name__ == "__main__":
    main()
