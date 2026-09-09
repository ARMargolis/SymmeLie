"""Finite-dimensional real Lie algebras; [e_a, e_b] = c[a,b,k] e_k."""
import torch
from torch import nn


class LieAlgebra(nn.Module):
    def __init__(self, structure_constants, *, atol=1e-8):
        super().__init__()
        c = torch.as_tensor(structure_constants).detach().clone()
        if c.dtype not in (torch.float32, torch.float64):
            raise ValueError("Use real float32 or float64 structure constants")
        if c.ndim != 3 or len(set(c.shape)) != 1:
            raise ValueError("Expected (dim, dim, dim) structure constants")
        if not torch.isfinite(c).all():
            raise ValueError("Structure constants must be finite")
        if not torch.allclose(c + c.transpose(0, 1), torch.zeros_like(c), atol=atol):
            raise ValueError("Bracket is not antisymmetric")
        jacobi = torch.einsum("abk,kcd->abcd", c, c)
        jacobi = jacobi + jacobi.permute(1, 2, 0, 3) + jacobi.permute(2, 0, 1, 3)
        if jacobi.numel() and jacobi.abs().max() > atol:
            raise ValueError("Bracket violates the Jacobi identity")
        self.register_buffer("structure_constants", c)

    @property
    def dim(self):
        return self.structure_constants.shape[0]

    def bracket(self, x, y):
        return torch.einsum("...a,...b,abk->...k", x, y, self.structure_constants)

    def adjoint(self):
        from .representation import Representation
        if self.dim == 0:
            raise ValueError("A discrete group's zero-dimensional algebra has no nonempty adjoint features")
        return Representation(self, self.structure_constants.transpose(1, 2))


def orthogonal(p, q=0, *, dtype=torch.float64, device=None):
    """Return (so(p,q), defining representation), metric diag(-I_p,+I_q).

    q=0 gives a compact orthogonal algebra. so(1,n) acts on H^n.
    """
    from .representation import Representation
    if not isinstance(p, int) or not isinstance(q, int) or min(p, q) < 0 or p + q < 2:
        raise ValueError("p,q must be nonnegative integers with p+q >= 2")
    n = p + q
    signs = [-1] * p + [1] * q
    basis = []
    for i in range(n):
        for j in range(i + 1, n):
            g = torch.zeros(n, n, dtype=dtype, device=device)
            g[i, j], g[j, i] = 1, -signs[i] * signs[j]
            basis.append(g)
    generators = torch.stack(basis)
    commutators = generators[:, None] @ generators[None, :] - generators[None, :] @ generators[:, None]
    # These generators are Frobenius-orthogonal with squared norm two.
    c = torch.einsum("abij,kij->abk", commutators, generators) / 2
    algebra = LieAlgebra(c)
    return algebra, Representation(algebra, generators)
