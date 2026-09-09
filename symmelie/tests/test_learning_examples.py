"""Behavioral checks for the independent tutorial adaptations."""
import itertools
import math
import pytest
import torch
from symmelie import orthogonal_group
from examples.inertia_tensor import InertiaNet, inertia_target, run as run_inertia
from examples.particle_forces import ParticleForceNet, run as run_forces
from examples.set_regression import SetRegressor, run as run_sets
from examples.tree_embedding import TreeEmbedding, tree_distances, embedding_stress, run as run_tree


@pytest.fixture(autouse=True)
def small_deterministic_cpu_workload():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(12)
    yield
    torch.set_num_threads(threads)


def test_inertia_tensor_symmetry_and_known_target():
    points = torch.randn(2, 5, 3, dtype=torch.float64)
    masses = torch.rand(2, 5, dtype=points.dtype) + .2
    model = InertiaNet()
    # Nonzero random coefficients prevent vacuous zero-map equivariance tests.
    with torch.no_grad():
        model.coupling.linear.coefficients.normal_()
    _, rep = orthogonal_group(3)
    q = rep.exp(points.new_tensor([.2, -.3, .7])) @ rep.discrete_generators[0]
    permutation = [2, 4, 0, 3, 1]
    moved = (points @ q.T + points.new_tensor([2., -1., 3.]))[:, permutation]
    result = model(points, masses)
    torch.testing.assert_close(model(moved, masses[:, permutation]), q @ result @ q.T)
    simple = points.new_tensor([[[-1., 0., 0.], [1., 0., 0.]]])
    expected = torch.diag(points.new_tensor([0., 2., 2.])).unsqueeze(0)
    torch.testing.assert_close(inertia_target(simple, masses.new_ones(1, 2)), expected)


@pytest.mark.parametrize("nodes", [3, 6])
def test_particle_forces_geometry_permutation_and_input_gradients(nodes):
    points = torch.randn(2, nodes, 3, dtype=torch.float64, requires_grad=True)
    masses = torch.rand(2, nodes, dtype=points.dtype) + .2
    model = ParticleForceNet()
    _, rep = orthogonal_group(3)
    q = rep.exp(points.new_tensor([-.4, .7, .1])) @ rep.discrete_generators[0]
    permutation = torch.randperm(nodes)
    out = model(points, masses)
    transformed = model((points @ q.T + 3)[:, permutation], masses[:, permutation])
    torch.testing.assert_close(transformed, (out @ q.T)[:, permutation])
    torch.testing.assert_close(out.sum(-2), torch.zeros_like(out[:, 0]), atol=1e-12, rtol=0)
    torch.testing.assert_close(model.dot(points, points).squeeze(-1), points.square().sum(-1))
    out.square().mean().backward()
    assert torch.isfinite(points.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.radial.parameters())


def test_set_regressor_all_permutations():
    model = SetRegressor()
    values = torch.randn(3, 4, dtype=torch.float64)
    expected = model(values)
    for permutation in itertools.permutations(range(4)):
        torch.testing.assert_close(model(values[:, list(permutation)]), expected)
    with pytest.raises(ValueError, match="4 scalar"):
        model(torch.randn(2, 5, dtype=values.dtype))


def test_tree_graph_and_embedding_gradients():
    distances = tree_distances(2)
    assert distances.shape == (7, 7)
    assert distances[3, 4] == 2 and distances[3, 6] == 4
    torch.testing.assert_close(distances, distances.T)
    torch.testing.assert_close(distances.diag(), torch.zeros(7, dtype=distances.dtype))
    model = TreeEmbedding(7)
    loss = embedding_stress(model, distances * .5)
    loss.backward()
    assert torch.isfinite(model.spatial.grad).all()
    assert model.spatial.grad.abs().max() > 0


@pytest.mark.parametrize("run,steps", [(run_inertia, 100), (run_forces, 100), (run_sets, 150)])
def test_training_improves_unseen_batch(run, steps):
    results = run(steps, seed=0)
    assert all(math.isfinite(value) for value in results.values())
    assert results['final_test_mse'] < results['initial_test_mse'] * .5
    for name, value in results.items():
        if name.endswith('max_error'):
            assert value < 1e-9


def test_tree_training_and_lorentz_invariance():
    results = run_tree(150, seed=0)
    assert all(math.isfinite(value) for value in results.values())
    assert results['final_reconstruction_stress'] < .5 * results['initial_reconstruction_stress']
    assert results['lorentz_distance_max_error'] < 1e-9
    assert results['manifold_max_error'] < 1e-10
    assert results['max_poincare_radius'] < 1
