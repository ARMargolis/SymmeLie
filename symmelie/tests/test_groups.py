import io
import math
import pytest
import torch
from symmelie import (LieAlgebra, Representation, EquivariantLinear, EquivariantBilinear,
                    intertwiner_basis, so, se, translations, torus, similarity, symplectic,
                    cyclic, dihedral, symmetric, orthogonal_group, octahedral)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(23)


@pytest.mark.parametrize("factory,n,algebra_dim,rep_dim", [
    (so, 2, 1, 2), (so, 3, 3, 3), (so, 4, 6, 4),
    (se, 2, 3, 3), (se, 3, 6, 4), (translations, 3, 3, 4),
    (similarity, 3, 7, 4), (symplectic, 1, 3, 2), (symplectic, 2, 10, 4), (torus, 2, 2, 4),
])
def test_continuous_factories(factory, n, algebra_dim, rep_dim):
    algebra, rep = factory(n)
    assert algebra.dim == algebra_dim and rep.dim == rep_dim
    features = rep.direct_sum(rep.trivial())
    layer = EquivariantLinear(features, features)
    x = torch.randn(4, features.dim, dtype=torch.float64)
    t = torch.randn(algebra.dim, dtype=x.dtype) * .2
    torch.testing.assert_close(layer(features.act(t, x)), features.act(t, layer(x)), atol=1e-9, rtol=1e-9)
    layer(x).square().sum().backward()
    assert torch.isfinite(layer.coefficients.grad).all()


def test_so4_preserves_metric_and_orientation():
    _, rep = so(4)
    rotation = rep.exp(torch.randn(6, dtype=torch.float64))
    torch.testing.assert_close(rotation.T @ rotation, torch.eye(4, dtype=rotation.dtype))
    torch.testing.assert_close(torch.linalg.det(rotation), rotation.new_tensor(1.))


def test_torus_independent_periodic_angles():
    _, rep = torus(2)
    t = torch.tensor([math.pi / 2, math.pi], dtype=torch.float64)
    x = torch.tensor([1., 0., 1., 0.], dtype=t.dtype)
    torch.testing.assert_close(rep.act(t, x), t.new_tensor([0., 1., -1., 0.]), atol=1e-12, rtol=0)
    torch.testing.assert_close(rep.act(t + 2 * math.pi, x), rep.act(t, x), atol=1e-12, rtol=0)


@pytest.mark.parametrize("n", [2, 3])
def test_rigid_points_directions_and_positive_rotation(n):
    algebra, rep = se(n)
    nr = n * (n - 1) // 2
    t = torch.zeros(algebra.dim, dtype=torch.float64)
    t[nr:] = torch.arange(1, n + 1, dtype=t.dtype)
    point = torch.cat((torch.arange(n, dtype=t.dtype), t.new_ones(1)))
    direction = point.clone()
    direction[-1] = 0
    torch.testing.assert_close(rep.act(t, point)[:n], point[:n] + t[nr:])
    torch.testing.assert_close(rep.act(t, direction), direction)
    turn = torch.zeros_like(t)
    turn[0 if n == 2 else 2] = math.pi / 2
    x = torch.zeros(n + 1, dtype=t.dtype)
    x[0] = 1
    expected = torch.zeros_like(x)
    expected[1] = 1
    torch.testing.assert_close(rep.act(turn, x), expected, atol=1e-12, rtol=0)


def test_se3_twists_wrenches_and_adjoint_conjugation():
    algebra, points = se(3)
    twists = algebra.adjoint()
    wrenches = twists.dual()
    t = torch.randn(6, dtype=torch.float64) * .3
    velocity, force = torch.randn(2, 6, dtype=t.dtype)
    pose = points.exp(t)
    hat = lambda x: torch.einsum("a,aij->ij", x, points.generators)
    torch.testing.assert_close(hat(twists.act(t, velocity)), pose @ hat(velocity) @ torch.linalg.inv(pose))
    torch.testing.assert_close((twists.act(t, velocity) * wrenches.act(t, force)).sum(), (velocity * force).sum())
    # se(3) twist bracket in standard [omega, v] convention.
    a, b = torch.randn(2, 6, dtype=t.dtype)
    expected = torch.cat((torch.linalg.cross(a[:3], b[:3]),
                          torch.linalg.cross(a[:3], b[3:]) - torch.linalg.cross(b[:3], a[3:])))
    torch.testing.assert_close(algebra.bracket(a, b), expected)


def test_similarity_and_symplectic_forms():
    _, rep = similarity(3)
    t = torch.zeros(7, dtype=torch.float64)
    t[-1] = math.log(2)
    torch.testing.assert_close(rep.exp(t), torch.diag(t.new_tensor([2., 2., 2., 1.])))
    _, rep = symplectic(2)
    t = torch.randn(10, dtype=torch.float64) * .2
    eye, zero = torch.eye(2, dtype=t.dtype), torch.zeros(2, 2, dtype=t.dtype)
    j = torch.cat((torch.cat((zero, eye), 1), torch.cat((-eye, zero), 1)), 0)
    matrix = rep.exp(t)
    torch.testing.assert_close(matrix.T @ j @ matrix, j)


@pytest.mark.parametrize("factory,order", [(cyclic, 1), (cyclic, 2), (cyclic, 5),
    (dihedral, 2), (dihedral, 4), (symmetric, 1), (symmetric, 4)])
def test_discrete_composition_and_layers(factory, order):
    algebra, rep = factory(order)
    assert algebra.dim == 0
    torch.testing.assert_close(rep.exp(torch.empty(0, dtype=torch.float64)), torch.eye(rep.dim, dtype=torch.float64))
    features = rep.direct_sum(rep.trivial(2))
    linear = EquivariantLinear(features, features)
    bilinear = EquivariantBilinear(rep, rep, rep.trivial())
    x, y = torch.randn(2, 4, rep.dim, dtype=torch.float64)
    z = torch.randn(4, features.dim, dtype=x.dtype)
    # Check a word, not just individual generators.
    transformed_x, transformed_y, transformed_z = x, y, z
    expected = linear(z)
    for index in list(range(rep.discrete_generators.shape[0])) * 3:
        transformed_x = rep.act_discrete(index, transformed_x)
        transformed_y = rep.act_discrete(index, transformed_y)
        transformed_z = features.act_discrete(index, transformed_z)
        expected = features.act_discrete(index, expected)
    torch.testing.assert_close(linear(transformed_z), expected, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(bilinear(transformed_x, transformed_y), bilinear(x, y))
    dual = rep.dual()
    for index in range(rep.discrete_generators.shape[0]):
        torch.testing.assert_close((dual.act_discrete(index, x) * rep.act_discrete(index, y)).sum(-1), (x*y).sum(-1))
    linear(z).square().sum().backward()
    bilinear(x, y).square().sum().backward()
    assert torch.isfinite(linear.coefficients.grad).all()
    assert torch.isfinite(bilinear.linear.coefficients.grad).all()


def test_finite_relations_and_invariant_dimensions():
    for n in (2, 3, 7):
        _, rep = dihedral(n)
        r, s = rep.discrete_generators
        eye = torch.eye(2, dtype=r.dtype)
        torch.testing.assert_close(torch.linalg.matrix_power(r, n), eye)
        torch.testing.assert_close(s @ s, eye)
        torch.testing.assert_close(s @ r @ s, torch.linalg.inv(r))
    _, c = cyclic(4)
    _, d = dihedral(4)
    assert intertwiner_basis(c, c).shape[0] == 2
    assert intertwiner_basis(d, d).shape[0] == 1  # reflection removes the skew map
    _, p = symmetric(4)
    assert intertwiner_basis(p, p).shape[0] == 2  # identity and all-ones
    assert intertwiner_basis(p.trivial(), p).shape[0] == 1


@pytest.mark.parametrize("reflections,size", [(False, 24), (True, 48)])
def test_cube_group_order(reflections, size):
    _, rep = octahedral(reflections=reflections)
    identity = torch.eye(3, dtype=torch.float64)
    key = lambda a: tuple(a.flatten().tolist())
    elements = {key(identity): identity}
    pending = [identity]
    while pending:
        current = pending.pop()
        for g in rep.discrete_generators:
            product = current @ g
            k = key(product)
            if k not in elements:
                elements[k] = product
                pending.append(product)
        assert len(elements) <= size
    assert len(elements) == size
    layer = EquivariantLinear(rep, rep)
    x = torch.randn(3, dtype=torch.float64)
    for matrix in elements.values():
        torch.testing.assert_close(layer(matrix @ x), matrix @ layer(x))


def test_continuous_and_disconnected_constraints():
    _, connected = so(2)
    _, full = orthogonal_group(2)
    assert intertwiner_basis(connected, connected).shape[0] == 2
    assert intertwiner_basis(full, full).shape[0] == 1
    layer = EquivariantLinear(full, full)
    x = torch.randn(3, 2, dtype=torch.float64)
    t = torch.tensor([.7], dtype=x.dtype)
    action = lambda z: full.act_discrete(0, full.act(t, z))
    torch.testing.assert_close(layer(action(x)), action(layer(x)))
    tensor = full.tensor_product(full)
    pair = (x.unsqueeze(-1)*x.unsqueeze(-2)).flatten(-2)
    transformed = (action(x).unsqueeze(-1)*action(x).unsqueeze(-2)).flatten(-2)
    torch.testing.assert_close(tensor.act_discrete(0, tensor.act(t, pair)), transformed)


def test_custom_nonorthogonal_discrete_dual():
    algebra, rep = cyclic(2)
    d = torch.tensor([[[1., 2.], [0., -1.]]], dtype=torch.float64)
    custom = Representation(algebra, rep.generators, discrete_generators=d, symmetry_id="custom-involution")
    x, y = torch.randn(2, 2, dtype=d.dtype)
    torch.testing.assert_close((custom.act_discrete(0, x)*custom.dual().act_discrete(0, y)).sum(), (x*y).sum())


def test_empty_constraints_and_discrete_validation():
    algebra = LieAlgebra(torch.empty(0, 0, 0, dtype=torch.float64))
    g = torch.empty(0, 2, 2, dtype=torch.float64)
    unconstrained = Representation(algebra, g)
    assert intertwiner_basis(unconstrained, unconstrained).shape == (4, 2, 2)
    with pytest.raises(ValueError, match="budget"):
        intertwiner_basis(unconstrained, unconstrained, max_entries=1)
    with pytest.raises(ValueError, match="symmetry_id"):
        Representation(algebra, g, discrete_generators=torch.eye(2, dtype=g.dtype).unsqueeze(0))
    with pytest.raises(ValueError, match="invertible"):
        Representation(algebra, g, discrete_generators=torch.zeros(1, 2, 2, dtype=g.dtype), symmetry_id="bad")
    with pytest.raises(ValueError, match="symmetry_id"):
        cyclic(4)[1].direct_sum(cyclic(5)[1])
    with pytest.raises(ValueError, match="budget"):
        EquivariantBilinear(cyclic(4)[1], cyclic(4)[1], cyclic(4)[1].trivial(), max_entries=1)


def test_discrete_dtype_and_state_dict():
    _, rep = dihedral(4, dtype=torch.float32)
    layer = EquivariantLinear(rep, rep)
    x = torch.randn(5, 2)
    torch.testing.assert_close(layer(rep.act_discrete(1, x)), rep.act_discrete(1, layer(x)))
    rep.double()
    assert rep.discrete_generators.dtype == torch.float64
    state = io.BytesIO()
    torch.save(rep.state_dict(), state)
    state.seek(0)
    _, restored = dihedral(4)
    restored.load_state_dict(torch.load(state, weights_only=True))
    torch.testing.assert_close(restored.discrete_generators, rep.discrete_generators)


@pytest.mark.parametrize("factory,n", [(so,1), (se,1), (translations,0), (symplectic,0),
                                      (cyclic,0), (dihedral,1), (symmetric,0)])
def test_invalid_dimensions(factory, n):
    with pytest.raises(ValueError):
        factory(n)
