"""Representations in a shared, ordered Lie algebra basis."""
import torch
from torch import nn


class Representation(nn.Module):
    def __init__(self, algebra, generators, *, discrete_generators=None, symmetry_id=None, atol=1e-7):
        super().__init__()
        g = torch.as_tensor(generators).detach().clone()
        c = algebra.structure_constants
        if g.ndim != 3 or g.shape[0] != algebra.dim or g.shape[1] != g.shape[2] or g.shape[1] == 0:
            raise ValueError("Expected generators shaped (algebra.dim, dim, dim)")
        if g.dtype != c.dtype or g.device != c.device or not torch.isfinite(g).all():
            raise ValueError("Generators must be finite and match algebra dtype/device")
        lhs = g[:, None] @ g[None, :] - g[None, :] @ g[:, None]
        rhs = torch.einsum("abk,kij->abij", c, g)
        if not torch.allclose(lhs, rhs, atol=atol, rtol=atol):
            raise ValueError("Generators do not represent the supplied Lie bracket")
        d = g.new_empty(0, g.shape[-1], g.shape[-1]) if discrete_generators is None else torch.as_tensor(discrete_generators).detach().clone()
        if d.ndim != 3 or d.shape[1:] != g.shape[1:]:
            raise ValueError("Expected discrete generators shaped (count, dim, dim)")
        if d.dtype != g.dtype or d.device != g.device or not torch.isfinite(d).all():
            raise ValueError("Discrete generators must be finite and match dtype/device")
        if d.shape[0] and (not isinstance(symmetry_id, str) or not symmetry_id):
            raise ValueError("Discrete generators require a nonempty symmetry_id identifying their ordered presentation")
        if d.shape[0] and torch.any(torch.linalg.slogdet(d).sign == 0):
            raise ValueError("Discrete generators must be invertible")
        self.algebra = algebra
        self.symmetry_id = symmetry_id
        self.register_buffer("generators", g)
        self.register_buffer("discrete_generators", d)

    @property
    def dim(self):
        return self.generators.shape[-1]

    def compatible(self, other):
        a, b = self.algebra.structure_constants, other.algebra.structure_constants
        if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device or not torch.allclose(a, b):
            raise ValueError("Representations must use the same ordered algebra basis, dtype and device")
        if self.symmetry_id != other.symmetry_id or self.discrete_generators.shape[0] != other.discrete_generators.shape[0]:
            raise ValueError("Representations must use the same symmetry_id and ordered discrete generators")

    def _new(self, generators, discrete_generators):
        return Representation(self.algebra, generators, discrete_generators=discrete_generators,
                              symmetry_id=self.symmetry_id)

    def act_discrete(self, index, features):
        """Apply one discrete generator; compose calls for a group word."""
        return torch.einsum("ij,...j->...i", self.discrete_generators[index], features)

    def exp(self, coordinates):
        """Batched group action exp(sum_a coordinates[a] generators[a])."""
        return torch.linalg.matrix_exp(torch.einsum("...a,aij->...ij", coordinates, self.generators))

    def act(self, coordinates, features):
        return torch.einsum("...ij,...j->...i", self.exp(coordinates), features)

    def trivial(self, dim=1):
        if not isinstance(dim, int) or dim < 1:
            raise ValueError("dim must be a positive integer")
        identity = torch.eye(dim, dtype=self.generators.dtype, device=self.generators.device)
        return self._new(self.generators.new_zeros(self.algebra.dim, dim, dim),
                         identity.expand(self.discrete_generators.shape[0], dim, dim))

    def dual(self):
        return self._new(-self.generators.transpose(-1, -2),
                         torch.linalg.inv(self.discrete_generators).transpose(-1, -2))

    def direct_sum(self, other):
        self.compatible(other)
        def combine(a, b):
            if a.shape[0] == 0:
                return a.new_empty(0, self.dim + other.dim, self.dim + other.dim)
            return torch.stack([torch.block_diag(x, y) for x, y in zip(a, b)])
        return self._new(combine(self.generators, other.generators),
                         combine(self.discrete_generators, other.discrete_generators))

    def tensor_product(self, other):
        self.compatible(other)
        i = torch.eye(self.dim, dtype=self.generators.dtype, device=self.generators.device)
        j = torch.eye(other.dim, dtype=i.dtype, device=i.device)
        continuous = torch.stack([
            torch.kron(a.contiguous(), j) + torch.kron(i, b.contiguous())
            for a, b in zip(self.generators, other.generators)
        ]) if self.algebra.dim else i.new_empty(0, self.dim * other.dim, self.dim * other.dim)
        discrete = torch.stack([
            torch.kron(a.contiguous(), b.contiguous())
            for a, b in zip(self.discrete_generators, other.discrete_generators)
        ]) if self.discrete_generators.shape[0] else continuous.new_empty(0, self.dim * other.dim, self.dim * other.dim)
        return self._new(continuous, discrete)
