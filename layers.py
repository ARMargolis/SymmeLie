"""Dense constraint-based equivariant layers for small representations."""
import math
import torch
from torch import nn


def _nullspace(matrix, rtol):
    # Construction is numerical, offline, and deliberately outside autograd.
    with torch.no_grad():
        _, s, vh = torch.linalg.svd(matrix, full_matrices=matrix.shape[0] < matrix.shape[1])
        rank = int((s > rtol * s.max()).sum()) if s.numel() else 0
        return vh[rank:].T.contiguous()


def intertwiner_basis(rep_in, rep_out, *, rtol=None, max_entries=10_000_000):
    """Basis B[k,out,in] solving G_out W = W G_in for every generator.

    Dense SVD is intended for small representations, not high tensor orders.
    """
    rep_in.compatible(rep_out)
    a, b = rep_in.generators, rep_out.generators
    ni, no = rep_in.dim, rep_out.dim
    generator_count = a.shape[0] + rep_in.discrete_generators.shape[0]
    if max(1, generator_count) * (ni * no) ** 2 > max_entries:
        raise ValueError("Dense constraint budget exceeded; use smaller representations")
    rtol = (1e-6 if a.dtype == torch.float32 else 1e-10) if rtol is None else rtol
    if not 0 < rtol < 1:
        raise ValueError("rtol must lie between zero and one")
    ii = torch.eye(ni, dtype=a.dtype, device=a.device)
    io = torch.eye(no, dtype=a.dtype, device=a.device)
    pairs = list(zip(a, b)) + list(zip(rep_in.discrete_generators, rep_out.discrete_generators))
    constraints = torch.cat([
        torch.kron(go.contiguous(), ii) - torch.kron(io, gi.T.contiguous())
        for gi, go in pairs
    ]) if pairs else a.new_empty(0, ni * no)
    return _nullspace(constraints, rtol).T.reshape(-1, no, ni)


class EquivariantLinear(nn.Module):
    def __init__(self, rep_in, rep_out, bias=True, **solver_options):
        super().__init__()
        basis = intertwiner_basis(rep_in, rep_out, **solver_options)
        self.register_buffer("basis", basis)
        self.coefficients = nn.Parameter(basis.new_empty(basis.shape[0]))
        nn.init.normal_(self.coefficients, std=1 / math.sqrt(max(1, basis.shape[0])))
        if bias:
            b = intertwiner_basis(rep_in.trivial(), rep_out, **solver_options).squeeze(-1)
            self.register_buffer("bias_basis", b)
            self.bias_coefficients = nn.Parameter(b.new_zeros(b.shape[0]))
        else:
            self.register_buffer("bias_basis", None)
            self.register_parameter("bias_coefficients", None)

    @property
    def weight(self):
        return torch.einsum("k,koi->oi", self.coefficients, self.basis)

    def forward(self, x):
        result = x @ self.weight.T
        if self.bias_basis is not None:
            result = result + self.bias_coefficients @ self.bias_basis
        return result


class EquivariantBilinear(nn.Module):
    """Learnable intertwiner from a tensor product, analogous to CG coupling."""
    def __init__(self, left, right, rep_out, **solver_options):
        super().__init__()
        left.compatible(right)
        # Reject large products before allocating their representation matrices.
        size = left.dim * right.dim
        budget = solver_options.get("max_entries", 10_000_000)
        if max(1, left.algebra.dim + left.discrete_generators.shape[0]) * size ** 2 > budget:
            raise ValueError("Tensor representation budget exceeded")
        self.linear = EquivariantLinear(left.tensor_product(right), rep_out, bias=False, **solver_options)

    def forward(self, x, y):
        return self.linear((x.unsqueeze(-1) * y.unsqueeze(-2)).flatten(-2))


class ScalarGate(nn.Module):
    """Multiply vectors by sigmoid gates. Caller must supply invariant scalars."""
    def forward(self, features, invariant_scalars):
        return features * torch.sigmoid(invariant_scalars).unsqueeze(-1)
