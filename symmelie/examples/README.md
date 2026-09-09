# Geometric deep learning with SymmELie

Four small, runnable adaptations of published geometric learning ideas. They are
independently written educational implementations with synthetic data, not copies
of upstream code, full reproductions, or performance comparisons with the papers.
Each script uses SymmELie's actual representations, layers, or manifold operations.

## Run locally

From the repository root, after installing SymmELie:

```bash
python -m pip install -e '.[test]'
python -m examples.inertia_tensor
python -m examples.particle_forces
python -m examples.set_regression
python -m examples.tree_embedding
```

No dataset download, GPU, plotting library, or graph library is needed. The CLIs
use float64 on CPU and one CPU thread for these small problems. They accept
`--steps`, `--seed`, and optional `--output path/to/metrics.json`. Run them from
the source checkout: `examples` is not part of the installed library distribution.
The model classes can be imported from this checkout for experimentation.

## 1. Predicting an inertia tensor

**Sources:** [Finzi, Welling & Wilson, EMLP (ICML 2021)](https://proceedings.mlr.press/v139/finzi21a.html)
and the authors' [Inertia dataset implementation](https://github.com/mfinzi/equivariant-MLP/blob/master/emlp/datasets.py).

**Run:** `python -m examples.inertia_tensor` (200 steps by default).

The task is to predict a 3x3 inertia tensor from five positive point masses and
their 3D positions. The model centers positions at the center of mass, applies
`EquivariantBilinear(vector, vector, vector.tensor_product(vector))` to each point,
and sums the resulting tensors weighted by mass. The generator solver finds the
permitted O(3) couplings; training learns their coefficients.

The target is `sum_i m_i (||r_i||^2 I - r_i r_i^T)`. Only data generation uses
that formula. The model starts with zero coefficients and learns the coupling.
Rotating or reflecting the input gives `I -> Q I Q^T`; centering also makes the
prediction invariant to translation and summation removes point ordering.

**Adaptation:** the upstream task uses inertia about the supplied origin and a
full EMLP. This version uses the center of mass, total mass normalized to one in
the generator, and a single polynomial coupling. It is deliberately a small,
representable physics task, not a test of a large deep architecture. Inputs are
`points: (batch,nodes,3)` and `masses: (batch,nodes)`; all masses must be positive.

## 2. Learning particle forces with radial messages

**Sources:** [Satorras, Hoogeboom & Welling, EGNN (ICML 2021)](https://proceedings.mlr.press/v139/satorras21a.html)
and the authors' [minimal EGNN implementation](https://github.com/vgsatorras/egnn/blob/main/models/egnn_clean/egnn_clean.py).

**Run:** `python -m examples.particle_forces` (250 steps).

`ParticleForceNet` constructs relative displacement vectors, obtains squared
distances through a calibrated and frozen SymmELie invariant bilinear map, and
uses a scalar MLP to predict pair weights from distance and mass product. A sum
of weighted displacements produces forces. This follows the invariant-scalar /
equivariant-vector message pattern in EGNN. The synthetic target is a softened
attractive pair force with strength `m_i*m_j/(1+distance_squared)^1.5`, averaged
over neighbors including a zero self-message.

**Adaptation:** one dense force-prediction block, no learned node updates,
velocities, integration, sparse edges, or original N-body dataset. Outputs are
force vectors, so they rotate/reflect and stay unchanged under translation;
they do not acquire the translation of a point. The symmetric pair weights
also make total predicted force zero to numerical precision. This does not
by itself give an energy-conserving numerical trajectory integrator.

Inputs match the inertia example. The block works with different node counts
without rebuilding its representations; its pairwise computation has O(N^2) cost.

## 3. Learning a statistic of an unordered set

**Source:** [Zaheer et al., Deep Sets (NeurIPS 2017)](https://arxiv.org/abs/1703.06114).

**Run:** `python -m examples.set_regression` (250 steps).

`SetRegressor` uses `symmetric(4)` and a tensor product with trivial channels to
construct two permutation-equivariant linear layers. Coordinatewise SiLU is
valid here because the representation only permutes channels. Mean pooling and
a scalar readout produce a permutation-invariant output. The target is the
mean squared value plus 0.3 times the mean, with inputs sampled in [-1,1].

**Adaptation:** illustrates equivariant set features followed by invariant
pooling. The dense S4 constraint solver fixes the set cardinality at construction;
this is not the variable-cardinality shared-MLP implementation commonly used
for Deep Sets. No padding/masking is supported. Default input shape: `(batch,4)`.
Using arbitrary coordinatewise SiLU on SO(3) or Lorentz vectors would not preserve
their equivariance; the validity here is specific to permutations.

## 4. Embedding a hierarchy in hyperbolic space

**Source:** [Nickel & Kiela, Poincare Embeddings (NeurIPS 2017)](https://arxiv.org/abs/1705.08039).

**Run:** `python -m examples.tree_embedding` (350 steps).

Learn positions for a 15-node binary tree in H2, minimizing squared error between
hyperbolic distances and half the tree's shortest-path distances. SymmELie's
`Hyperboloid` computes the geometry and converts the learned points to the
Poincare disk. Only distinct pairs enter the stress loss; the reported Lorentz
check compares pairwise squared distances before and after a boost/rotation.

**Adaptation:** uses a Lorentz spatial-coordinate chart and ordinary Adam on
that chart, with an all-pairs distance stress objective. The original uses
Poincare embeddings, a different training objective, and Riemannian optimization.
Our chart guarantees valid upper-sheet points but its optimization trajectory
is not Lorentz-equivariant. The geometric distances are Lorentz-invariant.
This is an embedding-table example, not a graph convolution network or an
inductive encoder for new nodes. A finite tree need not embed without distortion
in this fixed two-dimensional space.

## Evaluation and interpretation

For inertia, forces, and sets, the scripts evaluate the untrained and trained
model on the same 128-example unseen batch, while training uses freshly sampled
32-example batches. The reported test MSE measures synthetic-distribution
generalization, not the source papers' benchmark performance. Separate tests
check reflections, translations, node permutations, gradients, and training
improvement. Inertia and force classes can process arbitrary node counts;
the set representation is fixed-size.

The tree uses all pairwise distances for training and reports **reconstruction
stress**, not held-out accuracy or link-prediction performance. An embedding
with low reconstruction error has not demonstrated generalization to new nodes.

The [results directory](results/) contains actual default-seed CPU runs generated
by these commands with `--output`. Symmetry errors should be close to floating-point
precision; convergence and exact numbers can differ with seed, PyTorch version,
and the numerical basis returned by SVD. Run `python -m pytest -q` to check the
examples together with the library tests.

Recorded with seed 0, PyTorch 2.8.0, float64, CPU:

| Example | Metric | Before training | After training | Symmetry max error |
| --- | --- | --- | --- | --- |
| Inertia tensor | Unseen-batch MSE | 0.07704 | 0.00000185 | 8.9e-16 |
| Particle forces | Unseen-batch MSE | 0.06898 | 0.0001149 | 1.1e-16 |
| Set regression | Unseen-batch MSE | 0.12058 | 0.0000899 | 7.2e-16 |
| Tree embedding | Training reconstruction stress | 3.10120 | 0.06811 | 8.3e-15 |

The rows use different targets and scales, so the error values are not comparable
across tasks. The tree's symmetry metric measures distance invariance, while the
other rows measure the trained model's equivariance or permutation invariance.
