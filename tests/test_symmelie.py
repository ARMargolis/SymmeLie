import io
import pytest
import torch
from symmelie import (LieAlgebra, Representation, orthogonal, EquivariantLinear,
                    EquivariantBilinear, Hyperboloid, HyperbolicAggregation,
                    minkowski_dot, intertwiner_basis)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(7)


@pytest.mark.parametrize("signature", [(3, 0), (1, 2), (1, 3), (2, 2)])
def test_algebra_and_linear_equivariance(signature):
    alg, v = orthogonal(*signature)
    adj = alg.adjoint()
    rep = v.direct_sum(v).direct_sum(v.trivial(2))
    layer = EquivariantLinear(rep, rep)
    x = torch.randn(5, rep.dim, dtype=torch.float64)
    t = torch.randn(alg.dim, dtype=torch.float64) * .3
    torch.testing.assert_close(layer(rep.act(t, x)), rep.act(t, layer(x)), atol=1e-9, rtol=1e-9)
    a, b = torch.randn(2, alg.dim, dtype=torch.float64)
    torch.testing.assert_close(torch.einsum("a,aij,j->i", a, adj.generators, b), alg.bracket(a, b))
    layer(x).square().mean().backward()
    assert torch.isfinite(layer.coefficients.grad).all()


def test_lorentz_tensor_product_and_dual():
    alg, v = orthogonal(1, 3)
    bilinear = EquivariantBilinear(v, v, v.trivial())
    assert bilinear.linear.basis.shape[0] == 1
    x, y = torch.randn(2, 6, 4, dtype=torch.float64)
    t = torch.randn(alg.dim, dtype=torch.float64) * .3
    torch.testing.assert_close(bilinear(v.act(t, x), v.act(t, y)), bilinear(x, y))
    torch.testing.assert_close((v.dual().act(t, x) * v.act(t, y)).sum(-1), (x * y).sum(-1))
    assert intertwiner_basis(v, v.trivial()).shape[0] == 0


def test_nonsemisimple_heisenberg():
    c = torch.zeros(3, 3, 3, dtype=torch.float64)
    c[0, 1, 2], c[1, 0, 2] = 1, -1
    algebra = LieAlgebra(c)
    generators = torch.zeros_like(c)
    generators[0, 0, 1] = generators[1, 1, 2] = generators[2, 0, 2] = 1
    rep = Representation(algebra, generators)
    layer = EquivariantLinear(rep, rep)
    x = torch.randn(4, 3, dtype=torch.float64)
    t = torch.tensor([.2, -.3, .1], dtype=torch.float64)
    torch.testing.assert_close(layer(rep.act(t, x)), rep.act(t, layer(x)))


@pytest.mark.parametrize("radius", [.5, 1., 2.])
def test_hyperbolic_geometry(radius):
    h = Hyperboloid(radius)
    x, y = h.from_spatial(torch.randn(2, 5, 3, dtype=torch.float64) * .2)
    torch.testing.assert_close(minkowski_dot(x, x), torch.full((5,), -radius ** 2, dtype=x.dtype))
    tangent = h.logmap(x, y)
    torch.testing.assert_close(minkowski_dot(x, tangent), torch.zeros(5, dtype=x.dtype), atol=1e-12, rtol=0)
    torch.testing.assert_close(h.expmap(x, tangent), y)
    torch.testing.assert_close(h.from_poincare(h.to_poincare(x)), x)
    _, rep = orthogonal(1, 3)
    t = torch.randn(6, dtype=x.dtype) * .3
    torch.testing.assert_close(h.squared_distance(rep.act(t, x), rep.act(t, y)), h.squared_distance(x, y))
    torch.testing.assert_close(h.expmap(rep.act(t, x), rep.act(t, tangent)), rep.act(t, y))


def test_geometry_gradients_at_origin_and_away():
    h = Hyperboloid()
    for scale in (0., .2):
        s = (torch.randn(2, 3, dtype=torch.float64) * scale).requires_grad_()
        origin = h.from_spatial(torch.zeros_like(s))
        def operation(z):
            v = torch.cat((torch.zeros_like(z[..., :1]), z), -1)
            return h.expmap(origin, v)
        assert torch.autograd.gradcheck(operation, (s,))
        assert torch.autograd.gradcheck(lambda z: h.logmap(origin, h.from_spatial(z)), (s,))
        assert torch.autograd.gradcheck(lambda z: h.squared_distance(origin, h.from_spatial(z)), (s,))


def test_aggregation_equivariance_gradients_and_serialization():
    h = Hyperboloid()
    _, rep = orthogonal(1, 3)
    s = (torch.randn(2, 5, 3, dtype=torch.float64) * .2).requires_grad_()
    x = h.from_spatial(s)
    model = HyperbolicAggregation().double()
    t = torch.randn(6, dtype=x.dtype) * .3
    out = model(x)
    torch.testing.assert_close(model(rep.act(t, x)), rep.act(t, out), atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(minkowski_dot(out, out), -torch.ones_like(out[..., 0]))
    permutation = torch.tensor([2, 0, 4, 1, 3])
    torch.testing.assert_close(model(x[:, permutation]), out[:, permutation])
    h.squared_distance(out, x).mean().backward()
    assert torch.isfinite(s.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = HyperbolicAggregation().double()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(restored(x), out)


def test_validation_and_budget():
    with pytest.raises(ValueError):
        LieAlgebra(torch.ones(2, 2, 2, dtype=torch.float64))
    _, rep = orthogonal(1, 2)
    with pytest.raises(ValueError):
        Representation(rep.algebra, torch.randn_like(rep.generators))
    with pytest.raises(ValueError, match="budget"):
        intertwiner_basis(rep, rep, max_entries=1)
    with pytest.raises(ValueError):
        Hyperboloid(-1)
    with pytest.raises(ValueError):
        Hyperboloid().from_poincare(torch.ones(3))


def test_float32_and_module_conversion():
    _, rep = orthogonal(1, 2, dtype=torch.float32)
    model = EquivariantLinear(rep, rep)
    t = torch.tensor([.1, .2, -.1])
    x = torch.randn(5, 3)
    torch.testing.assert_close(model(rep.act(t, x)), rep.act(t, model(x)), atol=1e-5, rtol=1e-5)
    assert model.double()(x.double()).dtype == torch.float64
