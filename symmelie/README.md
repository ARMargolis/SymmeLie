# SymmELie

SymmELie is a small PyTorch research foundation for equivariant deep learning with supplied
finite-dimensional **real Lie algebra and discrete group representations**, including
rigid motions, finite symmetries, and the Lorentz symmetries of hyperbolic space.
This is an initial implementation, not a complete
general-purpose replacement for e3nn.

## About the name

**SymmELie** combines three overlapping references: **Sym** for **symmetry**,
**ymmE** for **Emmy** spelled backwards, and **Lie** for **Sophus Lie**.

**Emmy Noether** is honored for revealing the connection between continuous
symmetries of an action and conservation laws through Noether's theorem. Her
work motivates the idea that symmetry is a fundamental organizing principle
for physical models, and therefore a meaningful structure to build into neural
networks. The tribute does not imply that every SymmELie model automatically
satisfies a conservation law.

**Sophus Lie** is honored for developing the theory of continuous transformation
groups. Lie groups and Lie algebras provide the mathematical foundation for
SymmELie's generator-based representations and equivariant layers, including
those used for rotations, rigid motions, and hyperbolic geometry.

## Installation and example

Requires Python 3.9+.

The project is named **SymmELie**, with package name and Python import namespace
`symmelie`. Run the installation command from the project root.

```bash
python -m pip install -e '.[test]'
pytest -q
python -m examples.hyperbolic
python -m examples.robotics
```

```python
import torch
from symmelie import orthogonal, EquivariantBilinear, Hyperboloid, HyperbolicAggregation

algebra, vector = orthogonal(1, 3)  # so(1,3), default float64
scalar = vector.trivial()
invariant = EquivariantBilinear(vector, vector, scalar)
x = torch.randn(10, 4, dtype=torch.float64)
scores = invariant(x, x)  # learnable multiple of the Minkowski quadratic form

geometry = Hyperboloid(radius=1.0)  # sectional curvature -1
points = geometry.from_spatial(torch.randn(2, 12, 3, dtype=torch.float64) * .2)
layer = HyperbolicAggregation().double()
output = layer(points)  # (2, 12, 4), points remain on the upper hyperboloid
loss = geometry.squared_distance(output, points).mean()
loss.backward()
```

## Available symmetries and when to use them

For runnable training applications, see the [geometric deep learning examples](examples/README.md):
inertia-tensor prediction, EGNN-style particle forces, permutation-invariant set
regression, and hyperbolic tree embeddings. Each includes primary-source links,
an explanation of the adaptation, and an offline synthetic dataset.

Every constructor returns `(algebra, representation)` and accepts `dtype` and
`device`. Constructors default to float64. Group names below identify the action;
continuous generators represent the corresponding lowercase Lie algebra.

| Symmetry | Constructor | Representation / useful applications |
| --- | --- | --- |
| SO(2), SO(3), SO(4), SO(n) | `so(n)` | n-vector rotations. Planar headings, spatial vectors, or four-dimensional rotational symmetry. SO(4) has six generators. |
| SE(2), SE(3), SE(n) | `se(n)` | Homogeneous rigid motions; planar navigation, poses, manipulation, and frame changes. |
| Translations R^n | `translations(n)` | Homogeneous translations; position offsets and origin changes. |
| SO(2)^k torus | `torus(k)` | Independent 2D angle pairs; periodic joint features when independent angle-shift equivariance is appropriate. |
| Sim(n) | `similarity(n)` | Positive uniform scale plus rigid motion; similarity-frame and scale-equivariant models. |
| Sp(2n,R) | `symplectic(n)` | Real 2n-dimensional phase space, n canonical position/momentum pairs; linear canonical transformations. |
| SO(p,q), connected action | `orthogonal(p,q)` | Indefinite metric symmetries; `orthogonal(1,n)` for hyperbolic geometry. |
| Full O(n) | `orthogonal_group(n)` | Rotations plus reflection; use only when reflected configurations are also valid symmetries. |
| Cyclic C_m | `cyclic(m)` | Planar rotations by 2*pi/m; discrete orientation classes. |
| Dihedral D_m (2m elements) | `dihedral(m)` | Planar rotations and reflection; polygonal or mirrored planar symmetry. |
| Permutations S_m | `symmetric(m)` | Permutes m feature slots; exchangeable agents, sensors, or equivalent components. |
| Cube rotations / full cube group | `octahedral()` / `octahedral(reflections=True)` | 24 rotations / 48 symmetries on 3D vectors; cubic objects and axis-aligned discrete symmetries. |

For robotics, start with SE(2)/SE(3) and their rotation subgroups; add finite
symmetries when they match the robot or task. These choices are consistent with
[GTSAM's SE(3) pose representation](https://borglab.github.io/gtsam/pose3/),
[SE(2)-equivariant grasp learning](https://www.roboticsproceedings.org/rss18/p071.html),
and [finite morphological symmetries in robotics](https://roboticsproceedings.org/rss19/p053.html).
A fixed gravity direction, joint limits, obstacles, or asymmetric tools can reduce
the actual symmetry: encoding a configuration with a group does not establish
that the whole task is equivariant under it.

SO(4) was already expressible as `orthogonal(4)` and now has the named `so(4)`
entry point with tests. It describes rotations of four-dimensional vectors;
it is not a replacement for SO(3) just because a quaternion has four coordinates.
SU(2)/Spin groups and their spinorial representations are not implemented.

`symplectic(n)` means the **real** algebra sp(2n,R), not the compact quaternionic
algebra sp(n). Equivariance to symplectic transformations does not make a learned
layer symplectic or a numerical integrator energy-preserving. Likewise,
`EquivariantLinear` on homogeneous features need not return a point with final
coordinate one. The supplied group actions preserve the relevant geometry;
arbitrary learned intertwiners need not.

### Robotics: points, twists, and wrenches

```python
from symmelie import se, EquivariantLinear

algebra, points = se(3)
twists = algebra.adjoint()
wrenches = twists.dual()
layer = EquivariantLinear(twists, twists)

change_of_frame = torch.tensor([.2, -.1, .3, 1., 2., -.5], dtype=torch.float64)
velocity = torch.randn(8, 6, dtype=torch.float64)
transformed_velocity = twists.act(change_of_frame, velocity)
prediction = layer(transformed_velocity)
```

`se(3)` coordinates and twist features are ordered `[omega_x,omega_y,omega_z,
v_x,v_y,v_z]`, with standard cross-product rotation matrices. Dual wrench features
are `[torque_x,torque_y,torque_z,force_x,force_y,force_z]`; the dot product of a
twist and wrench is invariant under matching adjoint/coadjoint transformations.
`se(2)` uses `[theta,v_x,v_y]` with counterclockwise positive theta.
These are exponential coordinates: when rotation is nonzero, `v` is not simply
the final translation column of the exponentiated matrix.

Homogeneous points are `[x,y,z,1]`; free directions are `[vx,vy,vz,0]`.
For `se(n)` with n>3, rotations follow the lexicographic `E_ij-E_ji` convention
of `so(n)`, followed by translations. `similarity(n)` appends one log-scale
generator to the `se(n)` basis.

### Discrete and disconnected groups

Discrete groups are not additional nonzero Lie algebras. Their zero-dimensional
Lie algebra cannot distinguish one finite group from another. We store actual
transformation matrices `discrete_generators` and impose
`D_out[s] W = W D_in[s]` for each one, alongside any continuous constraints.
This also supports disconnected groups such as O(n), using a reflection together
with the continuous rotation generators.

```python
from symmelie import dihedral, symmetric, EquivariantLinear

_, planar = dihedral(4)  # eight elements: 90-degree rotations and reflections
layer = EquivariantLinear(planar, planar)
x = torch.randn(10, 2, dtype=torch.float64)
rotated = planar.act_discrete(0, x)
reflected = planar.act_discrete(1, x)

_, slots = symmetric(4)
four_slots_three_channels = slots.tensor_product(slots.trivial(3))
# Flatten (4, 3) slot/channel features to 12 for this representation.
```

`cyclic(m)` uses planar rotation, so `cyclic(2)` is a half-turn, not a mirror
reflection. `symmetric(2)` swaps two slots. A robot-specific bilateral symmetry
often needs both joint permutations and sign changes; provide those matrices
explicitly rather than treating either factory as a universal joint model.
`dihedral(m)` orders rotation then reflection across the x axis. `symmetric(m)`
orders adjacent swaps. `octahedral()` uses quarter-turns about z then x, with
optional reflection in coordinate zero. Compose `act_discrete` calls to apply
group words; no enumeration is required to construct layers.

For custom discrete representations, use `Representation(algebra, generators,
discrete_generators=matrices, symmetry_id="my-ordered-presentation")`; a purely
discrete algebra has constants of shape `(0,0,0)` and continuous generators of
shape `(0,feature_dim,feature_dim)`. All input/output representations must use
the same group presentation, ordered generators, and `symmetry_id`. The constructor
checks shape, finite values, and invertibility, but does not infer group relations,
finiteness, or mixed continuous/discrete conjugation relations. Those are the
caller's responsibility for custom data. The solver nevertheless enforces the
supplied matrices, including all their products.

Direct sums, duals, tensor products, trivial channels, linear biases, and bilinear
couplings all carry the discrete action. Discrete tensor products use `D1 ⊗ D2`,
whereas infinitesimal tensor generators use `G1 ⊗ I + I ⊗ G2`. Applying only the
infinitesimal formula to a finite group would give incorrect constraints.

## Mathematical scope

Define an algebra by structure constants `c[a,b,k]`, with
`[e_a,e_b] = sum_k c[a,b,k] e_k`. `LieAlgebra` checks antisymmetry and the Jacobi
identity. A `Representation` takes matrices `G[a]` and checks
`[G[a],G[b]] = sum_k c[a,b,k] G[k]`. The algebra also supplies its adjoint
representation. Representations support direct sums, duals, tensor products,
trivial channels, and differentiable matrix-exponential actions.

Input and output representations must refer to the **same ordered algebra
basis**. Structural equality cannot detect arbitrary relabelings that preserve
the structure constants; maintaining this convention is the caller's responsibility.

`EquivariantLinear` solves `G_out[a] W - W G_in[a] = 0` using a dense SVD.
Its trainable parameters combine a fixed basis of solutions. Biases live only
in the invariant output subspace. `EquivariantBilinear` applies the same solver
to a tensor product, providing learnable couplings without requiring an irrep
classification or a Clebsch–Gordan table. `ScalarGate` requires scalar inputs
that are invariant under the action; arbitrary feature channels are not valid gates.

These constraints imply equivariance under products of exponentials and supplied
discrete matrices, up to numerical tolerance. Disconnected symmetries such as
parity must be supplied explicitly (as `orthogonal_group` does). Lie algebra
constraints alone do not establish descent from the simply connected group to
every group with that Lie algebra. No automatic representation discovery or
irreducible decomposition is implemented. Nonsemisimple algebras are accepted;
the test suite includes the Heisenberg algebra.

## Hyperbolic geometry

Hyperbolic space is a homogeneous manifold, not itself a Lie algebra:
`H^n = SO^+(1,n) / SO(n)`. We use ambient coordinates with metric
`diag(-1,+1,...,+1)`, `x[0] > 0`, and `<x,x>_L = -radius^2`.
`orthogonal(1,n)` returns its isometry algebra and defining representation.
`orthogonal(p,q)` also supports other real orthogonal signatures; its convention
is `diag(-I_p,+I_q)`, so `orthogonal(3)` supplies compact so(3).

`Hyperboloid` supplies tangent projection, exponential/logarithmic maps,
distance/squared distance, and Poincare ball conversion (ball radius `radius`).
Operations assume valid manifold points and tangent vectors. `from_spatial` is
a coordinate chart, not an equivariant map from arbitrary Euclidean features.
The ordinary distance is nondifferentiable at coincidence; squared distance and
the maps use local series to support finite first derivatives there.

`HyperbolicAggregation` learns attention scores from invariant pairwise squared
distances, combines upper-sheet points with positive softmax weights, and
normalizes the result onto the hyperboloid. It is Lorentz- and permutation-equivariant.
Its normalized ambient centroid is **not** the intrinsic Frechet mean. It has
quadratic node cost and currently has no sparse graph or padding-mask interface.
All points must use the layer's radius.

Manifold-valued layers and vector-representation layers serve different purposes:
an arbitrary equivariant linear map need not preserve the hyperboloid. A generic
coordinatewise ReLU on Lorentz vectors does not preserve equivariance either.

## Numerical and practical limits

- Dense constraint construction has `(algebra_dim + discrete_count) * (dim_in * dim_out)^2`
  entries; a default 10-million-entry guard limits allocations. Dense SVD still
  becomes expensive well below large-model scale. Bilinear layers also guard
  tensor-representation allocation. With no generators, the full unconstrained
  basis is guarded instead. Sparse solvers and basis caching are future work.
- Numerical rank depends on generator scaling and the chosen relative tolerance
  (`1e-10` for float64, `1e-6` for float32). Near-degenerate or badly conditioned
  generators require special care. Bases are fixed, detached construction data;
  learning the generators through the nullspace solver is not supported.
- Use float64 for geometry and constraint construction. Large boosts and points
  near the Poincare boundary cause cancellation or overflow. This implementation
  does not promise reliable extreme-radius or extreme-distance calculations.
- Real float32/float64 representations are supported. Complex representations,
  infinite-dimensional representations, automatic irreps, convolution kernels,
  and Riemannian parameter optimizers are outside this initial scope.
- Parameters and fixed bases use standard PyTorch modules, buffers, `.to(...)`,
  and state dictionaries. Reconstruct a layer with the same representations
  and matching symmetry presentation before loading its state. `symmetry_id` is
  constructor metadata, not a saved tensor. GPU execution has not been verified.

## Design references

- [e3nn tensor products](https://docs.e3nn.org/en/stable/api/o3/o3_tp.html):
  representation-typed learnable couplings for O(3).
- [Finzi, Welling & Wilson, 2021](https://arxiv.org/abs/2104.09459):
  generator-constraint construction of equivariant networks for matrix groups.
- [Hyperbolic Neural Networks](https://arxiv.org/abs/1805.09112):
  neural operations in hyperbolic geometry.
- [Fully Hyperbolic Neural Networks](https://arxiv.org/abs/2105.14686):
  Lorentz-model neural architectures.
- [PyTorch matrix exponential](https://docs.pytorch.org/docs/stable/generated/torch.linalg.matrix_exp.html).

SymmELie is an independent implementation of elementary generator constraints
and Lorentz-model formulas, not a reproduction of these libraries or architectures.
